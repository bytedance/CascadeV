import argparse
import pickle
from pathlib import Path
import torch
from diffusion.model.tvae import tvae_create, tvae_decode
from cv_utils import visualize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=str, required=True)
    parser.add_argument('--vid', type=str, required=True)
    parser.add_argument('--fps', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    vae = tvae_create(device)

    lat = pickle.loads(Path(args.lat).read_bytes()).to(device)
    b, c, f, h, w = lat.shape

    print('Latent Shape:', lat.shape)
    slice_size = 4 if f != 25 else 5
    with torch.no_grad():
        vid = tvae_decode(vae, lat, slice_size, show_prog=True)
    print('Video Shape:', vid.shape)

    visualize([vid], args.vid, '', fps=args.fps, quality=7)
    print(f'Saved video to {args.vid}')
