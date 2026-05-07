import json
import math
import os
from itertools import islice

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

import time
from collections import defaultdict, OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx


class BaseGPUCacheManager:
    def __init__(self, vram_thresh_gb: float = 18.5):
        self.limit_bytes = int(vram_thresh_gb * (1024 ** 3))
        self.cache: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        self.entry_sizes: Dict[int, int] = {}
        self.current_bytes = 0
        self.transfer_stream = torch.cuda.Stream()

    def _entry_bytes(self, entry: Dict[str, Optional[torch.Tensor]]) -> int:
        return sum(
            v.element_size() * v.numel()
            for v in entry.values()
            if isinstance(v, torch.Tensor)
        )

    def _on_hit(self, view_id: int):
        return None

    def _on_insert(self, view_id: int):
        return None

    def _on_remove(self, view_id: int, entry_size: int):
        return None

    def _select_evict_id(self) -> int:
        raise NotImplementedError

    def _reset_policy_state(self):
        return None

    def _remove(self, view_id: int):
        if view_id not in self.cache:
            return
        entry_size = self.entry_sizes.pop(view_id)
        self.current_bytes -= entry_size
        self.cache.pop(view_id)
        self._on_remove(view_id, entry_size)

    def _store(self, view_id: int, entry: Dict[str, Optional[torch.Tensor]]):
        entry_bytes = self._entry_bytes(entry)
        if entry_bytes > self.limit_bytes:
            return
        if view_id in self.cache:
            self._remove(view_id)
        while self.current_bytes + entry_bytes > self.limit_bytes and self.cache:
            self._remove(self._select_evict_id())
        if self.current_bytes + entry_bytes > self.limit_bytes:
            return
        self.cache[view_id] = entry
        self.entry_sizes[view_id] = entry_bytes
        self.current_bytes += entry_bytes
        self._on_insert(view_id)

    def get_cache(
        self, view_id: int, compute_func: Callable[[], Dict[str, Optional[torch.Tensor]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        if view_id in self.cache:
            self._on_hit(view_id)
            return self.cache[view_id]
        new_entry = compute_func()
        self._store(view_id, new_entry)
        return new_entry

    def has_view(self, view_id: int) -> bool:
        return view_id in self.cache

    def can_prefetch(self, high_watermark: float = 0.9) -> bool:
        watermark = max(0.0, min(1.0, high_watermark))
        return self.current_bytes < int(self.limit_bytes * watermark) # stop before cache flaps

    def prefetch(
        self, view_id: int, compute_func: Callable[[], Dict[str, Optional[torch.Tensor]]]
    ) -> bool:
        if not self.can_prefetch():
            return False
        if view_id in self.cache:
            return False
        with torch.cuda.stream(self.transfer_stream):
            new_entry = compute_func()
        old_bytes = self.current_bytes
        self._store(view_id, new_entry)
        return (view_id in self.cache) and (self.current_bytes >= old_bytes)

    def purge_all(self):
        self.cache.clear()
        self.entry_sizes.clear()
        self.current_bytes = 0
        self._reset_policy_state()
        torch.cuda.empty_cache()


class GPULRUCacheManager(BaseGPUCacheManager):
    def __init__(self, vram_thresh_gb: float = 18.5):
        super().__init__(vram_thresh_gb=vram_thresh_gb)
        self.lru = OrderedDict()

    def _on_hit(self, view_id: int):
        self.lru.move_to_end(view_id)

    def _on_insert(self, view_id: int):
        self.lru[view_id] = None

    def _on_remove(self, view_id: int, entry_size: int):
        self.lru.pop(view_id, None)

    def _select_evict_id(self) -> int:
        return next(iter(self.lru))

    def _reset_policy_state(self):
        self.lru.clear()


class GPULFUCacheManager(BaseGPUCacheManager):
    def __init__(self, vram_thresh_gb: float = 18.5):
        super().__init__(vram_thresh_gb=vram_thresh_gb)
        self.freq: Dict[int, int] = {}
        self.last_touch: Dict[int, int] = {}
        self.touch_tick = 0

    def _mark_touched(self, view_id: int, bump_freq: bool):
        self.touch_tick += 1
        self.last_touch[view_id] = self.touch_tick
        if bump_freq:
            self.freq[view_id] = self.freq.get(view_id, 0) + 1

    def _on_hit(self, view_id: int):
        self._mark_touched(view_id, bump_freq=True)

    def _on_insert(self, view_id: int):
        self.freq[view_id] = 1
        self._mark_touched(view_id, bump_freq=False)

    def _on_remove(self, view_id: int, entry_size: int):
        self.freq.pop(view_id, None)
        self.last_touch.pop(view_id, None)

    def _select_evict_id(self) -> int:
        return min(
            self.cache.keys(),
            key=lambda vid: (self.freq.get(vid, 0), self.last_touch.get(vid, 0)),
        )

    def _reset_policy_state(self):
        self.freq.clear()
        self.last_touch.clear()
        self.touch_tick = 0


class GPUTwoQCacheManager(BaseGPUCacheManager):
    def __init__(self, vram_thresh_gb: float = 18.5, a1_fraction: float = 0.25):
        super().__init__(vram_thresh_gb=vram_thresh_gb)
        self.a1_fraction = max(0.05, min(0.9, a1_fraction))
        self.a1_queue = OrderedDict()
        self.am_lru = OrderedDict()
        self.tier: Dict[int, str] = {}
        self.a1_bytes = 0
        self.am_bytes = 0

    def _on_hit(self, view_id: int):
        tier = self.tier.get(view_id)
        if tier == "a1":
            self.a1_queue.pop(view_id, None)
            self.am_lru[view_id] = None
            self.tier[view_id] = "am"
            size = self.entry_sizes.get(view_id, 0)
            self.a1_bytes -= size
            self.am_bytes += size
        elif tier == "am":
            self.am_lru.move_to_end(view_id)

    def _on_insert(self, view_id: int):
        self.a1_queue[view_id] = None
        self.tier[view_id] = "a1"
        self.a1_bytes += self.entry_sizes.get(view_id, 0)

    def _on_remove(self, view_id: int, entry_size: int):
        tier = self.tier.pop(view_id, None)
        if tier == "a1":
            self.a1_queue.pop(view_id, None)
            self.a1_bytes -= entry_size
        elif tier == "am":
            self.am_lru.pop(view_id, None)
            self.am_bytes -= entry_size

    def _select_evict_id(self) -> int:
        a1_budget = int(self.limit_bytes * self.a1_fraction)
        if self.a1_queue and (self.a1_bytes > a1_budget or not self.am_lru):
            return next(iter(self.a1_queue))
        if self.am_lru:
            return next(iter(self.am_lru))
        return next(iter(self.a1_queue))

    def _reset_policy_state(self):
        self.a1_queue.clear()
        self.am_lru.clear()
        self.tier.clear()
        self.a1_bytes = 0
        self.am_bytes = 0


class GSCacheManager(GPULRUCacheManager):
    pass


def createGpuCacheManager(
    cache_mode: str, vram_thresh_gb: float, twoq_a1_ratio: float
) -> BaseGPUCacheManager:
    if cache_mode in ("lru", "warm_all"):
        return GPULRUCacheManager(vram_thresh_gb=vram_thresh_gb)
    if cache_mode == "lfu":
        return GPULFUCacheManager(vram_thresh_gb=vram_thresh_gb)
    if cache_mode == "twoq":
        return GPUTwoQCacheManager(
            vram_thresh_gb=vram_thresh_gb, a1_fraction=twoq_a1_ratio
        )
    raise ValueError(f"Unsupported cache mode: {cache_mode}")

@dataclass
class Config:
    disable_viewer: bool = False
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"

    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    load_exposure: bool = True

    port: int = 8080

    # batch_size: int = 2
    batch_size: int = 1
    steps_scaler: float = 1.0

    # max_steps: int = 20_000
    # max_steps: int = 40_000
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2

    near_plane: float = 0.01
    far_plane: float = 1e10

    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False

    random_bkgd: bool = False

    # optimizer_stride: int = 2
    optimizer_stride: int = 4
    enable_frustum_culling: bool = False
    frustum_cull_radius_scale: float = 1.5
    frustum_cull_margin_early: float = 0.0
    frustum_cull_margin_late: float = 0.0
    frustum_cull_margin_switch_step: int = 10_000
    frustum_cull_interval: int = 10
    cache_mode: Literal["none", "lru", "lfu", "twoq", "warm_all"] = "none"
    enable_input_cache: bool = False
    enable_prefetch: bool = False
    # prefetch_lookahead: int = 2
    prefetch_lookahead: int = 1
    twoq_a1_ratio: float = 0.25
    # vram_thresh_gb: float = 16.0
    vram_thresh_gb: float = 18.5

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    bilateral_grid_fused: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    ppisp_use_controller: bool = True
    ppisp_controller_distillation: bool = True
    ppisp_controller_activation_num_steps: int = 25_000
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    use_color_correction_metric: bool = False

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        # points = init_extent * scene_scale * torch.randn((init_num_pts, 3))
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = batch_size * world_size
    optimizer_class = None
    # optimizer_class = torch.optim.AdamW
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)

        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("model initialized. number of gs:", len(self.splats["means"]))

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        self._gaussians_frozen = False

        self._view_cull_masks: Dict[int, Tensor] = {}
        self._view_cull_mask_steps: Dict[int, int] = {}

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("distillation: gaussian parameters frozen")

    @torch.no_grad()
    def frustum_cull(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        step: int,
        near_plane: float = 0.01,
    ) -> Tensor:
        """Return a boolean mask [N] — True for Gaussians visible in at least one camera.

        Uses the conservative radius approach from the CLM paper (§5.1):
        project each Gaussian center into camera space, compute a pixel-space
        radius from the largest scale, and keep the Gaussian if the bounding
        circle overlaps the image for any camera in the batch.
        """
        means = self.splats["means"]
        scales = torch.exp(self.splats["scales"])
        max_scale = scales.max(dim=-1).values

        viewmats = torch.linalg.inv(camtoworlds)
        R = viewmats[:, :3, :3]
        t = viewmats[:, :3, 3]

        p_cam = (R @ means.T).permute(0, 2, 1) + t.unsqueeze(1)
        z = p_cam[..., 2]

        valid_depth = z > near_plane

        fx = Ks[:, 0, 0].unsqueeze(1)
        fy = Ks[:, 1, 1].unsqueeze(1)
        cx = Ks[:, 0, 2].unsqueeze(1)
        cy = Ks[:, 1, 2].unsqueeze(1)

        z_safe = z.clamp(min=1e-6)
        u = fx * p_cam[..., 0] / z_safe + cx
        v = fy * p_cam[..., 1] / z_safe + cy

        f_max = torch.max(fx, fy)

        radius_scale = max(self.cfg.frustum_cull_radius_scale, 0.0)
        r = radius_scale * f_max * max_scale.unsqueeze(0) / z_safe

        if step < self.cfg.frustum_cull_margin_switch_step:
            margin_scale = self.cfg.frustum_cull_margin_early
        else:
            margin_scale = self.cfg.frustum_cull_margin_late
        margin_scale = max(margin_scale, 0.0)
        e = margin_scale * max(width, height)
        in_frustum = (
            valid_depth
            & (u + r > -e) 
            & (u - r < width + e)
            & (v + r > -e) 
            & (v - r < height + e)
        )

        return in_frustum.any(dim=0)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        external_buffers=None,
        cull_mask: Optional[Tensor] = None,

        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        image_ids = kwargs.pop("image_ids", None)

        cull_idx = None
        if cull_mask is not None:
            cull_idx = cull_mask.nonzero(as_tuple=False).squeeze(1)
            means_in = means[cull_idx]
            quats_in = quats[cull_idx]
            scales_in = scales[cull_idx]
            opacities_in = opacities[cull_idx]
        else:
            means_in, quats_in, scales_in, opacities_in = means, quats, scales, opacities

        if external_buffers is not None:
            return (
                external_buffers["render_colors"],
                external_buffers["render_alphas"],
                external_buffers["info"],
            )

        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
            if cull_idx is not None:
                colors = colors[cull_idx]
        else:
            if cull_idx is not None:
                sh0_in = self.splats["sh0"][cull_idx]
                shN_in = self.splats["shN"][cull_idx]
                colors = torch.cat([sh0_in, shN_in], 1)
            else:
                colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means_in,
            quats=quats_in,
            scales=scales_in,
            opacities=opacities_in,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )

        if cull_idx is not None:
            info["global_gaussian_ids"] = cull_idx

        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)

            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def prepare_rasterization_buffers(self, step, camtoworlds, Ks, width, height, image_ids, camera_idcs, exposure, masks):
        sh_degree_to_use = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)
    
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree_to_use,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
            masks=masks,
            camera_idcs=camera_idcs,
            exposure=exposure,
        )

        return {
            "info": {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in info.items()},
            "render_colors": renders.detach(),
            "render_alphas": alphas.detach()
        }

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        cacheMode = cfg.cache_mode
        if cacheMode == "none" and cfg.enable_input_cache:
            cacheMode = "lru"
        cache_enabled = cacheMode != "none"
        prefetch_lookahead = max(int(cfg.prefetch_lookahead), 0)
        prefetch_enabled = (
            cache_enabled and cfg.enable_prefetch and prefetch_lookahead > 0
        )
        cache_manager: Optional[BaseGPUCacheManager] = None
        if cache_enabled:
            cache_manager = createGpuCacheManager(
                cache_mode=cacheMode,
                vram_thresh_gb=cfg.vram_thresh_gb,
                twoq_a1_ratio=cfg.twoq_a1_ratio,
            )
            self.cache_manager = cache_manager

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)
    
        max_steps = cfg.max_steps
        init_step = 0
        
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.post_processing == "bilateral_grid":
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=cfg.batch_size, 
            shuffle=True,
            num_workers=4, 
            persistent_workers=True, 
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        profile_start, profile_stop = 0, max_steps + 1

        optimizer_stride = cfg.optimizer_stride

        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        _last_loss: float = 0.0
        _prefetch_events: dict = {}
        def _next_train_batch():
            nonlocal trainloader_iter
            try:
                return next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                return next(trainloader_iter)

        def _to_device_batch(batch):
            return {
                "pixels": batch["image"].to(device, non_blocking=True).to(torch.uint8), # keep uint8 in cache for less mem
                "camtoworlds": batch["camtoworld"].to(device, non_blocking=True),
                "Ks": batch["K"].to(device, non_blocking=True),
                "image_ids": batch["image_id"].to(device, non_blocking=True),
                "camera_idcs": batch["camera_idx"].to(device, non_blocking=True),
                "masks": (
                    batch["mask"].to(device, non_blocking=True)
                    if batch.get("mask") is not None
                    else None
                ),
                "exposure": (
                    batch["exposure"].to(device, non_blocking=True)
                    if "exposure" in batch
                    else None
                ),
                "points": (
                    batch["points"].to(device, non_blocking=True)
                    if cfg.depth_loss
                    else None
                ),
                "depths_gt": (
                    batch["depths"].to(device, non_blocking=True)
                    if cfg.depth_loss
                    else None
                ),
            }

        queue_target = 1 + (prefetch_lookahead if prefetch_enabled else 0)
        data_queue = deque()

        def _fill_data_queue():
            while len(data_queue) < queue_target:
                data_queue.append(_next_train_batch())

        if cacheMode == "warm_all" and cache_manager is not None:
            warm_loader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=2,
                persistent_workers=False,
                pin_memory=True,
            )
            warm_iter = tqdm.tqdm(
                warm_loader,
                desc="Warm GPU cache",
                disable=world_rank != 0,
            )
            for warm_data in warm_iter:
                warm_view_id = int(warm_data["image_id"][0].item())
                cache_manager.prefetch(
                    warm_view_id,
                    lambda batch=warm_data: _to_device_batch(batch),
                )
            torch.cuda.current_stream().wait_stream(cache_manager.transfer_stream)


        _fill_data_queue()

        for step in pbar:
            if step == profile_start:
                profiler.start()

            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            nvtx.range_push(f"Iteration_{step}")

            nvtx.range_push("Data_Loading")
            data = data_queue.popleft()
            _fill_data_queue()

            view_id = int(data["image_id"][0].item())


            def _load_data(batch=data):
                return _to_device_batch(batch)

            if cache_manager is not None:
                cached_data = cache_manager.get_cache(view_id, _load_data)
            else:
                cached_data = _load_data()
            
            pixels = cached_data["pixels"]
            if pixels.dtype == torch.uint8:
                pixels = pixels.float() / 255.0
            else:
                pixels = pixels.float()
            camtoworlds = cached_data["camtoworlds"]
            Ks = cached_data["Ks"]
            image_ids = cached_data["image_ids"]
            camera_idcs = cached_data["camera_idcs"]
            masks = cached_data["masks"]
            exposure = cached_data["exposure"]

            if cfg.depth_loss:
                points = cached_data["points"]
                depths_gt = cached_data["depths_gt"]

            height, width = pixels.shape[1:3]


            camtoworlds_gt = camtoworlds
            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            if cache_manager is not None:
                evt = _prefetch_events.pop(view_id, None)
                if evt is not None:
                    evt.wait()

            if cache_manager is not None and prefetch_enabled:
                prefetched_this_step = 0
                queued_view_ids = set()
                # max_prefetch_per_step = 2
                max_prefetch_per_step = 1
                for _nd in islice(data_queue, prefetch_lookahead):
                    next_view_id = int(_nd["image_id"][0].item())
                    if next_view_id == view_id:
                        continue
                    if next_view_id in queued_view_ids:
                        continue
                    queued_view_ids.add(next_view_id)
                    if next_view_id in _prefetch_events:
                        continue
                    if cache_manager.has_view(next_view_id):
                        continue
                    # this is a tiny guard becuase churn gets bad fast
                    enqueued = cache_manager.prefetch(
                        next_view_id,
                        lambda batch=_nd: _to_device_batch(batch),
                    )
                    if not enqueued:
                        continue
                    _evt = torch.cuda.Event()
                    _evt.record(cache_manager.transfer_stream)
                    _prefetch_events[next_view_id] = _evt
                    prefetched_this_step += 1
                    if prefetched_this_step >= max_prefetch_per_step:
                        break

            nvtx.range_pop()

            nvtx.range_push("Forward_Rasterize")
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            WARMUP_STEPS = 4000
            culling_start_step = WARMUP_STEPS
            if isinstance(cfg.strategy, DefaultStrategy):
                culling_start_step = max(culling_start_step, cfg.strategy.refine_stop_iter)
            use_culling = cfg.enable_frustum_culling and (step >= culling_start_step)

            if use_culling:
                recalc_interval = max(int(cfg.frustum_cull_interval), 1)
                last_mask_step = self._view_cull_mask_steps.get(view_id, -10**9)
                needs_recompute = (
                    view_id not in self._view_cull_masks
                    or (step - last_mask_step) >= recalc_interval
                )
                if needs_recompute:
                    self._view_cull_masks[view_id] = self.frustum_cull(
                        camtoworlds=camtoworlds,
                        Ks=Ks,
                        width=width,
                        height=height,
                        step=step,
                        near_plane=cfg.near_plane,
                    )
                    self._view_cull_mask_steps[view_id] = step
                cull_mask = self._view_cull_masks[view_id]
            else:
                cull_mask = None


            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=camera_idcs,
                exposure=exposure,
                external_buffers=None, 
                cull_mask=cull_mask,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            nvtx.range_pop()

            nvtx.range_push("Strategy_Pre_Backward")
            self.cfg.strategy.step_pre_backward(
                params=self.splats, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=info,
            )
            nvtx.range_pop()
    
            nvtx.range_push("Loss_and_Backward")
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda # mix both so it can calcualte smoother updates
            if cfg.depth_loss:
                points_ndc = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points_ndc.unsqueeze(2)
                depths_sampled = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )
                depths_sampled = depths_sampled.squeeze(3).squeeze(1)
                disp = torch.where(
                    depths_sampled > 0.0,
                    1.0 / depths_sampled,
                    torch.zeros_like(depths_sampled),
                )
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            (loss / optimizer_stride).backward()
            nvtx.range_pop()

            if step % 10 == 0:
                _last_loss = loss.item()
            desc = f"loss={_last_loss:.3f}| sh degree={sh_degree_to_use}| " # keep pbar text short
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            nvtx.range_push("Strategy_Post_Backward")
         
            base_thresh = 0.0002
            previous_grow_grad2d = None
            if isinstance(cfg.strategy, DefaultStrategy):
                previous_grow_grad2d = cfg.strategy.grow_grad2d
                if step < WARMUP_STEPS:
                    cfg.strategy.grow_grad2d = base_thresh * 0.5
                else:
                    cfg.strategy.grow_grad2d = base_thresh

            if optimizer_stride > 1:
                for v in info.values():
                    if isinstance(v, torch.Tensor) and v.grad is not None:
                        v.grad.mul_(optimizer_stride)

            n_gaussians_before = len(self.splats["means"])
            cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
            if len(self.splats["means"]) != n_gaussians_before:
                self._view_cull_masks.clear()
                self._view_cull_mask_steps.clear()

            if optimizer_stride > 1:
                for v in info.values():
                    if isinstance(v, torch.Tensor) and v.grad is not None:
                        v.grad.zero_()

            if previous_grow_grad2d is not None:
                cfg.strategy.grow_grad2d = previous_grow_grad2d
            nvtx.range_pop()

            if (step+1) % optimizer_stride == 0:
                nvtx.range_push("Optimizer_Update")
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
                nvtx.range_pop()
    
            nvtx.range_pop()

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.detach(), step)
                self.writer.add_scalar("train/l1loss", l1loss.detach(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.detach(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.detach(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.detach(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("step:", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                if self.cfg.app_opt:
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]
                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            if step == profile_stop:
                profiler.stop()
                print(f"profiling complete. captured steps {profile_start} to {profile_stop}.")
                break
    
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"psnr: {stats['psnr']:.3f}, ssim: {stats['ssim']:.4f}, lpips: {stats['lpips']:.3f} "
                    f"cc_psnr: {stats['cc_psnr']:.3f}, cc_ssim: {stats['cc_ssim']:.4f}, cc_lpips: {stats['cc_lpips']:.3f} "
                    f"time: {stats['ellipse_time']:.3f}s/image "
                    f"number of gs: {stats['num_GS']}"
                )
            else:
                print(
                    f"psnr: {stats['psnr']:.3f}, ssim: {stats['ssim']:.4f}, lpips: {stats['lpips']:.3f} "
                    f"time: {stats['ellipse_time']:.3f}s/image "
                    f"number of gs: {stats['num_GS']}"
                )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("exporting ppisp reports...")

        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"ppisp reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("viewer running... ctrl+c to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
