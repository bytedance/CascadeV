set -ex

PROJ_ROOT=`dirname $0`
PROJ_ROOT=`readlink -f $PROJ_ROOT`

####################
## Install PixArt ##
####################

cd $PROJ_ROOT
if [ ! -d PixArt-sigma ]
then
    git clone https://github.com/PixArt-alpha/PixArt-sigma.git
    cd PixArt-sigma
    git checkout 716d75d7f59f3498ab4a712cdbd28b6a93b50174
    pip install -r requirements.txt
    pip install setuptools==69.5.1 imageio==2.31.2 imageio-ffmpeg==0.4.9
fi

####################
## Install PixArt ##
####################

CODE_ROOT=$PROJ_ROOT/PixArt-sigma
cp $PROJ_ROOT/configs/*.py $CODE_ROOT/configs/pixart_sigma_config
cp $PROJ_ROOT/patch/train.py $CODE_ROOT/train_scripts/train.py
cp $PROJ_ROOT/patch/InternalData.py $CODE_ROOT/diffusion/data/datasets/InternalData.py
cp $PROJ_ROOT/patch/PixArt.py $CODE_ROOT/diffusion/model/nets/PixArt.py
cp $PROJ_ROOT/patch/gaussian_diffusion.py $CODE_ROOT/diffusion/model/gaussian_diffusion.py
cp $PROJ_ROOT/patch/checkpoint.py $CODE_ROOT/diffusion/utils/checkpoint.py
cp $PROJ_ROOT/patch/dpm_solver.py $CODE_ROOT/diffusion/model/dpm_solver.py
cp $PROJ_ROOT/patch/tvae.py $CODE_ROOT/diffusion/model/tvae.py
cp $PROJ_ROOT/patch/evae.py $CODE_ROOT/diffusion/model/evae.py
cp $PROJ_ROOT/patch/cv_utils.py $CODE_ROOT/cv_utils.py
