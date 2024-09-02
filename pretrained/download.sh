set -ex

CURR_ROOT=`dirname $0`
CURR_ROOT=`readlink -f $CURR_ROOT`
cd $CURR_ROOT

wget -q --show-progress https://huggingface.co/stabilityai/StableWurst/resolve/main/previewer.safetensors
wget -q --show-progress https://huggingface.co/stabilityai/StableWurst/resolve/main/effnet_encoder.safetensors
wget -q --show-progress https://huggingface.co/ByteDance/CascadeV/resolve/main/s1024.effn-f32.pth
