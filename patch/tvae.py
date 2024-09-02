from argparse import ArgumentParser
import torch
from diffusers import AutoencoderKLTemporalDecoder
from tqdm import trange
import pickle
from pathlib import Path
from cv_utils import load_video, preprocess_video, visualize


def tvae_create(device):
    vae = AutoencoderKLTemporalDecoder.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', subfolder="vae", torch_dtype=torch.float16, variant="fp16")
    vae = vae.eval().to(device).half()
    return vae


def tvae_encode(vae, ori, slice_size=8, show_prog=True, sample_posterior=True):
    b, c, t, h, w = ori.shape
    assert t % slice_size == 0, (t, slice_size)
    n = b * t
    vid = ori.permute(0, 2, 1, 3, 4).reshape(n, c, h, w).half()
    num_slices = n // slice_size
    lat = []
    pbar = trange(num_slices) if show_prog else range(num_slices)
    for i in pbar:
        vid_slice = vid[slice_size*i:slice_size*(i+1)]
        if sample_posterior:
            lat_slice = vae.encode(vid_slice).latent_dist.sample()
        else:
            lat_slice = vae.encode(vid_slice).latent_dist.mode()
        lat.append(lat_slice)
    lat = torch.cat(lat, dim=0)
    n, c, h, w = lat.shape
    lat = lat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return lat


def tvae_decode(vae, ori, slice_size=8, show_prog=True):
    b, c, t, h, w = ori.shape
    if slice_size == 0: slice_size = t
    assert t % slice_size == 0, (t, slice_size)
    n = b * t
    lat = ori.permute(0, 2, 1, 3, 4).reshape(n, c, h, w).half()
    num_slices = n // slice_size
    vid = []
    pbar = trange(num_slices) if show_prog else range(num_slices)
    for i in pbar:
        lat_slice = lat[slice_size*i:slice_size*(i+1)]
        vid_slice = vae.decode(lat_slice, num_frames=slice_size).sample
        vid.append(vid_slice)
    vid = torch.cat(vid, dim=0)
    n, c, h, w = vid.shape
    vid = vid.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return vid


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-i', '--src', type=str, required=True)
    parser.add_argument('-o', '--dst', type=str, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--beg', type=int, default=0)
    parser.add_argument('--num', type=int, default=32)
    parser.add_argument('--lat', type=str, default='')
    parser.add_argument('--square', action='store_true')
    args = parser.parse_args()
    device = 'cuda'

    vid_ori, fps = load_video(args.src, args.size, args.beg, args.num, size_div=16) # FHWC -> NCFHW
    vid_ori = preprocess_video(vid_ori).to(device)
    if args.square:
        h, w = vid_ori.shape[-2:]
        s = min(h, w)
        l, t = [(v - s) // 2 for v in (w, h)]
        vid_ori = vid_ori[...,t:t+s,l:l+s]
    print('Video Ori:', vid_ori.shape, vid_ori.min(), vid_ori.max())

    vae = tvae_create(device)

    print('Encoding ...')
    with torch.no_grad():
        vid_lat = tvae_encode(vae, vid_ori)
    print('Video Lat:', vid_lat.shape, vid_lat.min(), vid_lat.max())

    if args.lat:
        Path(args.lat).write_bytes(pickle.dumps(vid_lat))
        print(f'Saved lats to {args.lat}')

    print('Decoding ...')
    with torch.no_grad():
        vid_rec = tvae_decode(vae, vid_lat)
    print('Video Rec:', vid_rec.shape, vid_rec.min(), vid_rec.max())

    visualize([vid_ori, vid_rec], args.dst, args.dst[:-4] + '.jpg', fps)
