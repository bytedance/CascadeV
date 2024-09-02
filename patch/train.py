# Modified from https://github.com/PixArt-alpha/PixArt-sigma/blob/master/train_scripts/train.py
import argparse
import datetime
import os
import time
import warnings
from pathlib import Path
import torch.nn.functional as F
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from mmcv.runner import LogBuffer
import numpy as np
from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.data.datasets.InternalData import load_frames, get_transforms
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import get_world_size
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.model.tvae import tvae_create, tvae_encode
from diffusion.model.evae import evae_create, evae_encode
from cv_utils import visualize
warnings.filterwarnings("ignore")


TRANSFORMS = get_transforms()


def downsample_spatial(vid, scale):
    b, c, t, h, w = vid.shape
    rsz = F.interpolate(vid.reshape(b*c*t, 1, h, w), size=(h//scale, w//scale), mode='bilinear', align_corners=False)
    return rsz.reshape(b, c, t, h//scale, w//scale)


def upsample_spatial(vid, scale):
    b, c, t, h, w = vid.shape
    rsz = F.interpolate(vid.reshape(b*c*t, 1, h, w), size=(h*scale, w*scale), mode='bilinear', align_corners=False)
    return rsz.reshape(b, c, t, h*scale, w*scale)


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()
    global_step = start_step + 1

    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            train_as_img = config.joint_training is not None and np.random.random() < float(config.joint_training)
            with torch.no_grad():

                ## Load video
                imgs = batch[0]
                b, c, h, w = imgs.shape
                h = h // config.num_frames_per_video
                x = imgs.reshape(b, c, config.num_frames_per_video, h, w)

                ## Crop images
                curr_size = image_size
                if config.multi_scale:
                    _, _, _, h, w = x.shape
                    assert image_size == 512 and min(w, h) >= 1024
                    curr_size = np.random.choice([512, 1024])
                    crop_size = np.random.randint(curr_size, min(w, h) + 1)
                    if crop_size < min(w, h):
                        l = np.random.randint(w-crop_size+1)
                        t = np.random.randint(h-crop_size+1)
                        x = x[:,:,:,t:t+crop_size,l:l+crop_size].clone()
                    if curr_size == 1024:
                        b, c, t, h, w = x.shape
                        nt = t // 4
                        x = x.reshape(b, c, 4, nt, h, w).permute(0, 2, 1, 3, 4, 5).reshape(b * 4, c, nt, h, w)

                ## Get target
                num_frames = x.shape[2]
                z = tvae_encode(tvae, x, 1, show_prog=False, sample_posterior=config.sample_posterior)

                ## Get condition
                if config.encoder == 'evae':
                    assert config.sr_scale % 4 == 0, config.sr_scale
                    if config.sr_scale != 4:
                        lr_x = downsample_spatial(x, config.sr_scale//4)
                        lr_z = evae_encode(evae, lr_x, num_frames, show_prog=False, sample_posterior=True)
                    else:
                        lr_z = evae_encode(evae, x, num_frames, show_prog=False, sample_posterior=True)
                    rsz_z = upsample_spatial(lr_z, config.sr_scale)
                else:
                    lr_x = downsample_spatial(x, config.sr_scale)
                    lr_z = tvae_encode(tvae, lr_x, 1, show_prog=False, sample_posterior=True)
                    rsz_z = upsample_spatial(lr_z, config.sr_scale)

                ## Crop latents (if needed)
                curr_lat_size = curr_size // 8
                _, _, _, h, w = z.shape
                assert h == w and h >= curr_lat_size
                num_crops = 0
                if h > curr_lat_size: num_crops = 1
                if config.multi_scale and curr_size == 512: num_crops = 4
                if num_crops > 0:
                    zs, rsz_zs = [], []
                    for _ in range(num_crops):
                        assert min(w, h) >= curr_lat_size
                        l = np.random.randint(w-curr_lat_size+1)
                        t = np.random.randint(h-curr_lat_size+1)
                        z_crop = z[:,:,:,t:t+curr_lat_size,l:l+curr_lat_size]
                        z_ref_crop = rsz_z[:,:,:,t:t+curr_lat_size,l:l+curr_lat_size]
                        zs.append(z_crop)
                        rsz_zs.append(z_ref_crop)
                    z = zs[0].clone() if len(zs) == 1 else torch.cat(zs, dim=0)
                    rsz_z = rsz_zs[0].clone() if len(rsz_zs) == 1 else torch.cat(rsz_zs, dim=0)

                ## Pretend to be images
                b, c, t, h, w = z.shape
                z = z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                b, c, t, h, w = rsz_z.shape
                rsz_z = rsz_z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                timesteps = torch.randint(0, config.train_sampling_steps, (b,), device=z.device).long()
                timesteps = torch.stack([timesteps] * t, dim=1).reshape(-1)

            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(model, z * config.scale_factor, rsz_z * config.scale_factor, timesteps, model_kwargs=dict(y=None, mask=None, data_info={}, train_as_img=train_as_img))
                loss = loss_term['loss'].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {'loss': accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step: {global_step}, total_eta: {eta}, " \
                    f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.8f}, "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()

            if global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    return args


def load_last_ckpt(root):
    ckpts = list(Path(root).glob('checkpoints/epoch_*_step_*.pth'))
    if not ckpts: return None
    ckpts_and_steps = []
    for ckpt in ckpts:
        step = int(ckpt.name[:-4].split('_')[-1])
        ckpts_and_steps.append((step, ckpt))
    ckpts_and_steps.sort(key=lambda x: x[0])
    last_ckpt = ckpts_and_steps[-1][1]
    print(f'Found last ckpt: {last_ckpt}')
    return str(last_ckpt)


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir

    last_ckpt = load_last_ckpt(config.work_dir)
    if last_ckpt is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=last_ckpt,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with='tensorboard',
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=None,
        even_batches=True,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    image_size = int(config.image_size)
    latent_size = image_size // 8

    # Loading VAEs
    assert config.encoder in ('tvae', 'evae'), config.encoder
    assert config.decoder in ('tvae',), config.decoder
    lats_per_vid = config.num_frames_per_video

    logger.info(f'Loading TVAE to {accelerator.device} ...')
    tvae = tvae_create(accelerator.device)
    tvae.requires_grad_(False)

    if config.encoder == 'evae' or config.decoder == 'evae':
        logger.info(f'Loading EVAE to {accelerator.device} ...')
        evae = evae_create(accelerator.device)
        evae['dec'].requires_grad_(False)
        evae['enc'].requires_grad_(False)

    # Build model
    in_channels = 20 if config.encoder == 'evae' else 8
    model_kwargs = {
        'in_channels': in_channels, "pe_interpolation": config.pe_interpolation, 
        'attn_strides': config.attn_strides, 'lats_per_vid': lats_per_vid
    }
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=True, pred_sigma=True, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=True,
                        pred_sigma=True,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False))
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # Build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, config=config)
    train_dataloader = build_dataloader(dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True)

    # Build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers('dit4v', tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch, start_step = 0, 0
    total_steps = len(train_dataloader) * config.num_epochs

    ## Resume training
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1]) - 1
        start_step = int(path.replace('.pth', '').split("_")[3])
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler
                                                 )
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    ## Train
    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()
