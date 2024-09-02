# Modified from https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/data/datasets/InternalData.py
import os
import json
import numpy as np
from torch.utils.data import Dataset
from diffusion.data.builder import DATASETS
from diffusion.utils.logger import get_root_logger
from pathlib import Path
import cv2 as cv
import torchvision.transforms as T


def get_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize([.5], [.5]),
    ])


def crop_pad_cvt(img, resolution, roi=None):
    h, w, c = img.shape
    assert c == 3

    ## Pad
    if min(w, h) < resolution:
        l, r = 0, 0
        if w < resolution:
            l = (resolution - w) // 2
            r = resolution - w - l
        if h < resolution:
            t = (resolution - h) // 2
            b = resolution - h - t
        assert min(l, t, r, b) >= 0 and max(l, t, r, b) > 0
        img = cv.copyMakeBorder(img, t, b, l, r, borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))
        h, w, _ = img.shape
    assert min(h, w) >= resolution

    ## Crop
    if roi is None:
        l, t = (w - resolution) // 2, (h - resolution) // 2
        r, b = l + resolution, t + resolution
        roi = (l, t, r, b)
    l, t, r, b = roi
    img = img[t:b,l:r]

    ## BGR -> RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img, roi


def load_images(path, resolution, num_frames, time_stride, offset=None):
    files = sorted(Path(path).glob('*.jpg'))
    max_frames = len(files)
    assert max_frames >= num_frames * time_stride, f'Video too short: {time_stride}x{num_frames} vs {max_frames}'

    if offset is None:
        offset = np.random.randint(max_frames - time_stride * num_frames + 1)

    imgs = []
    roi = None
    for i in range(num_frames):
        img_file = files[i * time_stride + offset]
        img = cv.imread(str(img_file))
        assert img is not None, f'Failed to read frame #{img_file}'
        img, roi = crop_pad_cvt(img, resolution, roi)
        imgs.append(img)
    return imgs


def load_frames(path, resolution, num_frames, time_stride, offset=None):
    if Path(path).is_dir(): return load_images(path, resolution, num_frames, time_stride, offset)

    cap = cv.VideoCapture(str(path))
    max_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    assert max_frames >= num_frames * time_stride, f'Video too short: {time_stride}x{num_frames} vs {max_frames}'

    if offset is None:
        offset = np.random.randint(max_frames - time_stride * num_frames + 1)

    rows = []
    roi = None
    for i in range(num_frames):
        fi = i * time_stride + offset
        cap.set(cv.CAP_PROP_POS_FRAMES, fi)
        ret, img = cap.read()
        assert ret and img is not None, f'Failed to read frame #{fi}'
        img, roi = crop_pad_cvt(img, resolution, roi)
        rows.append(img)

    cap.release()
    return rows

def replace_img_ext():
    pass

@DATASETS.register_module()
class InternalData(Dataset):
    pass

@DATASETS.register_module()
class InternalDataSigma(Dataset):
    def __init__(self, config, **kwargs):
        self.transform = get_transforms()
        self.resolution = int(config.sample_size)
        self.videos = []
        self.time_stride = int(config.time_stride)
        self.num_frames = int(config.num_frames_per_video)

        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f'Resolution: {self.resolution}')
        logger.info(f'Num frames per video: {self.num_frames}')
        logger.info(f'Frame sample interval: {self.time_stride}')
        for f in config.video_list:
            if f.endswith('.json'):
                samples = json.loads(Path(f).read_text())
                samples = [v['path'] for v in samples]
            else:
                samples = Path(f).read_text().splitlines()
            logger.info(f'==> Found {len(samples)} videos in {f}')
            self.videos.extend(samples)
        self.ori_imgs_nums = len(self.videos)
        logger.info(f"Found {self.ori_imgs_nums} videos in total")

    def getdata(self, index):
        imgs = load_frames(self.videos[index], self.resolution, self.num_frames, self.time_stride)
        vid = np.vstack(imgs)
        if np.random.random() > 0.5:
            vid = np.fliplr(vid).copy()
        vid = self.transform(vid)
        return vid, ''

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"[Data Error] {self.videos[idx]}: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def __len__(self):
        return len(self.videos)
