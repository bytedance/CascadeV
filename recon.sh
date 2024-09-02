set -ex

PROJ_ROOT=`dirname $0`
PROJ_ROOT=`readlink -f $PROJ_ROOT`
CODE_ROOT=$PROJ_ROOT/PixArt-sigma
export PYTHONPATH=$CODE_ROOT
cd $PROJ_ROOT


##################
## Model Config ##
##################

CONFIG=s1024.effn-f32
CKPT_FILE=$PROJ_ROOT/pretrained/s1024.effn-f32.pth


################
## I/O Config ##
################

IMG_SIZE=1024
FPS=8
NUM=8
INPUT_FILE=$PROJ_ROOT/example/demo.mp4
OUTPUT_DIR=$PROJ_ROOT/example/outputs/$CONFIG
mkdir -p $OUTPUT_DIR


####################
## Compress Video ##
####################

python3 $PROJ_ROOT/patch/evae.py \
    --size $IMG_SIZE --num $NUM \
    -i $INPUT_FILE -o $OUTPUT_DIR/enc.mp4 \
    --lat $OUTPUT_DIR/enc.pkl --square


####################
## Refine Latents ##
####################

python3 $PROJ_ROOT/recon/refine_latent.py \
    $CODE_ROOT/configs/pixart_sigma_config/$CONFIG.py \
    --seed 42 --step 8 --scheduler EulerA --ckpt $CKPT_FILE \
    --src_lat $OUTPUT_DIR/enc.pkl --dst_lat $OUTPUT_DIR/rec.pkl


####################
## Reconstruction ##
####################

python3 $PROJ_ROOT/recon/lat2vid.py \
    --lat $OUTPUT_DIR/rec.pkl --vid $OUTPUT_DIR/rec.mp4 \
    --fps $FPS


###################
## Visualization ##
###################

python3 $PROJ_ROOT/recon/view_multi.py --fps $FPS \
    --src $OUTPUT_DIR/enc.mp4 $OUTPUT_DIR/rec.mp4 \
    --dst $OUTPUT_DIR/cmp.mp4 --img $OUTPUT_DIR/cmp.png
