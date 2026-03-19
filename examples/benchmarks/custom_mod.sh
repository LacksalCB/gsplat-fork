SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark"
RENDER_TRAJ_PATH="ellipse"

# Parse arguments: --scenes takes the scene list; all other flags are forwarded to trainer.py
SCENES=""
EXTRA_TRAINER_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes)
            SCENES="$2"
            shift 2
            ;;
        *)
            EXTRA_TRAINER_ARGS="$EXTRA_TRAINER_ARGS $1"
            shift
            ;;
    esac
done

if [ -z "$SCENES" ]; then
    echo "Error: --scenes argument is required" >&2
    exit 1
fi

SCENE_LIST=($SCENES)

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python trainer.py default  --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --eval_steps 0 \
        $EXTRA_TRAINER_ARGS

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=0 python trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT \
            $EXTRA_TRAINER_ARGS
    done
done


for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done