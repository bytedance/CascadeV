import torch
import torch.nn.functional as F
import imageio
import numpy as np
import cv2 as cv
from PIL import Image


def rescale_and_cvt(img, short_edge, size_div):
    sh, sw, _ = img.shape
    scale = short_edge / min(sh, sw)
    dh, dw = [round(v * scale) for v in [sh, sw]]

    img = cv.blur(img, ksize=(3, 3))
    img = cv.resize(img, (dw, dh))

    ch, cw = [size_div * (v // size_div) for v in [dh, dw]]
    if ch != dh or cw != dw:
        l, t = (dw - cw) // 2, (dh - ch) // 2
        img = img[t:t+ch,l:l+cw]

    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def load_video(path, short_edge, frame_beg, frame_num, frame_stp=1, size_div=32):
    assert short_edge % size_div == 0
    cap = cv.VideoCapture(str(path))
    frame_max = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv.CAP_PROP_FPS))
    frame_end = frame_beg + (frame_num - 1) * frame_stp
    assert frame_max > frame_end, f'Video too short: {frame_end + 1} vs {frame_max}'

    frames = []
    for i in range(frame_num):
        fi = i * frame_stp + frame_beg
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, img = cap.read()
        assert ret and img is not None, f'Failed to read frame #{fi}'
        img = rescale_and_cvt(img, short_edge, size_div)
        frames.append(img)
    cap.release()

    frames = np.stack(frames, axis=0)
    return frames, round(fps / frame_stp)


def preprocess_video(video):
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float() # FHWC -> NCFHW
    return video / 127.5 - 1.


def upsample(vid, frames, height):
    b, c, f, h, w = vid.shape
    if height != h:
        assert height > h and height % h == 0, f'Height indivisible'
        s = height // h
        vid = F.interpolate(vid.reshape(b*c*f, 1, h, w), size=(h*s, w*s), mode='bilinear', align_corners=False)
        vid = vid.reshape(b, c, f, h*s, w*s)

    b, c, f, h, w = vid.shape
    if frames != f:
        assert frames > f and frames % f == 0, f'Num frames indivisible'
        s = frames // f
        vid = torch.stack([vid]*s, dim=3).reshape(b, c, f*s, h, w)
    return vid


def visualize(vids, vid_file, img_file, fps, quality=5, max_imgs=16):
    fs, hs = [], []
    for vid in vids:
        _, _, f, h, _ = vid.shape
        fs.append(f)
        hs.append(h)
    frames, height = max(fs), max(hs)

    rows = []
    for vid in vids:
        _, _, f, h, _ = vid.shape
        if f != frames or h != height:
            vid = upsample(vid, frames, height)
        vid = torch.clamp(127.5 * vid + 127.5, 0, 255).type(torch.uint8)
        vid = vid.squeeze(0).permute(1, 2, 3, 0)
        rows.append(vid.cpu())
    rows = torch.cat(rows, dim=2)

    if vid_file:
        if str(vid_file).lower().endswith('.gif'):
            imageio.mimwrite(vid_file, rows, duration=round(1000/fps), quality=quality)
        else:
            imageio.mimwrite(vid_file, rows, fps=fps, quality=quality)

    if img_file:
        if max_imgs > 0:
            rows = rows[:max_imgs]
        imgs = np.vstack(rows.numpy())
        imgs = Image.fromarray(imgs)
        imgs.save(img_file)
