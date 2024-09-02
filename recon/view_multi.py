import argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import cv2 as cv


def upsample(vid, t_scale, r_scale):
    vid = torch.from_numpy(vid).float().cuda()
    _, h, w, _ = vid.shape
    vid = F.interpolate(vid.permute(0, 3, 1, 2), size=(h*r_scale, w*r_scale), mode='bilinear', align_corners=False)
    vid = vid.permute(0, 2, 3, 1)
    if t_scale > 1:
        n, c, h, w = vid.shape
        vid = torch.stack([vid]*t_scale, dim=1).reshape(n*t_scale, c, h, w)
    vid = torch.clamp(vid, 0, 255).cpu().numpy().astype(np.uint8)
    return vid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, nargs='+')
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--fps', type=int, required=True)
    parser.add_argument('--img', type=str, default='')
    args = parser.parse_args()

    ns, hs, vids = [], [], []
    for f in args.src:
        vid = imageio.mimread(f, memtest=False)
        vid = np.stack(vid)
        n, h, _, _ = vid.shape
        vids.append(vid)
        ns.append(n)
        hs.append(h)

    num_frames = max(ns)
    height = max(hs)
    for i in range(len(vids)):
        n, h, _, _ = vids[i].shape
        assert num_frames % n == 0, (n, num_frames)
        t_scale = num_frames // n
        assert height % h == 0, (h, height)
        r_scale = height // h
        vids[i] = upsample(vids[i], t_scale, r_scale)
    vids = np.concatenate(vids, axis=2)
    print(f'Saving video to {args.dst} ...')
    imageio.mimwrite(args.dst, vids, fps=args.fps, quality=5)
    if args.img:
        print(f'Saving image to {args.img} ...')
        assert cv.imwrite(args.img, cv.cvtColor(vids[0], cv.COLOR_RGB2BGR))
