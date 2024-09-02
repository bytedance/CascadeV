set -ex

PROJ_ROOT=`dirname $0`
PROJ_ROOT=`readlink -f $PROJ_ROOT`
CODE_ROOT=$PROJ_ROOT/PixArt-sigma
export PYTHONPATH=$CODE_ROOT


############
## Config ##
############

CONFIG=s1024.effn-f32
NUM_GPUS=1
NUM_MACHINES=1
MASTER_PORT=12345


###########
## Train ##
###########

cd $PROJ_ROOT
cp $PROJ_ROOT/configs/$CONFIG.py $CODE_ROOT/configs/pixart_sigma_config/$CONFIG.py
python3 -m torch.distributed.launch \
    --nnodes=$NUM_MACHINES --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    $CODE_ROOT/train_scripts/train.py \
    $CODE_ROOT/configs/pixart_sigma_config/$CONFIG.py \
    --work-dir $PROJ_ROOT/ckpt/$CONFIG
