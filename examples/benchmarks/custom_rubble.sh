shopt -s nullglob
# Scenes are paths relative to the examples/ working directory, e.g. "data/rubble-colmap".
# Multiple scenes can be passed as a space-separated list in $1.
SCENE_LIST=($1)
RESULT_DIR=$2
if [ -z "$3" ]; then
	DEVICE=2
else
	DEVICE=$3
fi
EXTRA_ARGS="${@:4}"   # all args after $1, $2, and $3 passed through to trainer file

RENDER_TRAJ_PATH="ellipse"
DATA_FACTOR=4   # 4K source images (~4600x3400) → ~1150x850 at factor 4

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    # train without eval
    time CUDA_VISIBLE_DEVICES=$DEVICE python -u trainer_refactored.py default --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir $SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --eval_steps 0 \
        $EXTRA_ARGS

    # run eval and render
    echo "Running eval and render for $SCENE"
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        time CUDA_VISIBLE_DEVICES=$DEVICE python -u trainer_refactored.py default --disable_viewer --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir $SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT \
            $EXTRA_ARGS
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
