from argparse import ArgumentParser
import torch
import torchvision
from tqdm import trange
import pickle
from pathlib import Path
import safetensors
from torch import nn
from cv_utils import load_video, preprocess_video, visualize


## Copied from https://github.com/Stability-AI/StableCascade/blob/master/modules/effnet.py
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
        self.mapper = nn.Sequential(
            nn.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),
        )
    def forward(self, x):
        return self.mapper(self.backbone(x))


## Copied from https://github.com/Stability-AI/StableCascade/blob/master/modules/previewer.py
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)


def load_state_dict(model, ckpt):
    state_dict = {}
    with safetensors.safe_open(ckpt, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)


def evae_create(device):
    encoder = EfficientNetEncoder()
    load_state_dict(encoder, 'pretrained/effnet_encoder.safetensors')
    encoder.eval().to(device).requires_grad_(False).half()

    decoder = Previewer()
    load_state_dict(decoder, 'pretrained/previewer.safetensors')
    decoder.eval().to(device).requires_grad_(False).half()

    transforms = torchvision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    return {'enc': encoder, 'dec': decoder, 'T': transforms}


def evae_encode(vae, ori, slice_size=1, show_prog=True, sample_posterior=True):
    b, c, t, h, w = ori.shape
    assert t % slice_size == 0, (t, slice_size)
    n = b * t
    vid = ori.permute(0, 2, 1, 3, 4).reshape(n, c, h, w).half()
    vid = vae['T'](0.5 * (vid + 1))

    num_slices = n // slice_size
    lat = []
    pbar = trange(num_slices) if show_prog else range(num_slices)
    for i in pbar:
        vid_slice = vid[slice_size*i:slice_size*(i+1)]
        if sample_posterior: # TODO
            lat_slice = vae['enc'](vid_slice)
        else:
            lat_slice = vae['enc'](vid_slice)
        lat.append(lat_slice)
    lat = torch.cat(lat, dim=0)
    n, c, h, w = lat.shape
    lat = lat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return lat


def evae_decode(vae, ori, slice_size=1, show_prog=True):
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
        vid_slice = vae['dec'](lat_slice)
        vid.append(vid_slice)
    vid = torch.cat(vid, dim=0)
    n, c, h, w = vid.shape
    vid = vid.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    vid = torch.clamp(2. * vid  - 1., -1., 1.)
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

    vid_ori, fps = load_video(args.src, args.size, args.beg, args.num) # FHWC -> NCFHW
    vid_ori = preprocess_video(vid_ori).to(device)
    if args.square:
        h, w = vid_ori.shape[-2:]
        s = min(h, w)
        l, t = [(v - s) // 2 for v in (w, h)]
        vid_ori = vid_ori[...,t:t+s,l:l+s]
    print('Video Ori:', vid_ori.shape, vid_ori.min(), vid_ori.max())

    vae = evae_create(device)

    print('Encoding ...')
    with torch.no_grad():
        vid_lat = evae_encode(vae, vid_ori)
    print('Video Lat:', vid_lat.shape, vid_lat.min(), vid_lat.max())

    if args.lat:
        Path(args.lat).write_bytes(pickle.dumps(vid_lat))
        print(f'Saved lats to {args.lat}')

    print('Decoding ...')
    with torch.no_grad():
        vid_rec = evae_decode(vae, vid_lat)
    print('Video Rec:', vid_rec.shape, vid_rec.min(), vid_rec.max())

    visualize([vid_rec], args.dst, args.dst[:-4] + '.jpg', fps)
