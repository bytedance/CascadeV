import argparse
import sys
import pickle
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import torch
from diffusion import DPMS
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import load_checkpoint
from diffusion.utils.misc import set_random_seed, read_config
import torch.nn.functional as F
from tqdm import tqdm
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler


NAME2SCHED = {
    'DDIM': DDIMScheduler,
    'EulerA': EulerAncestralDiscreteScheduler,
}


def upsample(lat, scale):
    b, c, t, h, w = lat.shape
    rsz = F.interpolate(lat.reshape(b*c*t, 1, h, w), size=(h*scale, w*scale), mode='bilinear', align_corners=False)
    return rsz.reshape(b, c, t, h*scale, w*scale)


def denoise_with_diffuser_sched(sched, ldm, latents, steps, ref_z):
    scheduler = NAME2SCHED[sched]()
    scheduler.set_timesteps(steps, device=latents.device)
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma
    for t in tqdm(timesteps):
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        current_timestep = t
        current_timestep = current_timestep[None].to(latent_model_input.device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])

        noise_pred = ldm.forward_with_dpmsolver(
            torch.cat([latent_model_input, ref_z], dim=1),
            timestep=current_timestep,
            y=None, mask=None
        )
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--src_lat', type=str, required=True)
    parser.add_argument('--dst_lat', type=str, required=True)
    parser.add_argument('--scheduler', type=str, required=True)
    args = parser.parse_args()

    config = read_config(args.config)
    set_random_seed(args.seed)
    device = torch.device('cuda')
    latent_size = int(config.image_size) // 8

    if config.decoder == 'tvae':
        lats_per_vid = config.num_frames_per_video
    else:
        lats_per_vid = config.num_frames_per_video // 4

    print(f'Loading PixArt ...')
    in_channels = 20 if config.encoder == 'evae' else 8
    model_kwargs = {
        'in_channels': in_channels, 'pe_interpolation': config.pe_interpolation,
        'attn_strides': config.attn_strides, 'lats_per_vid': lats_per_vid
    }
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=True,
                        pred_sigma=True,
                        **model_kwargs).eval().to(device)
    missing, unexpected = load_checkpoint(args.ckpt, model)
    print(f'==> Missing keys: {missing}')
    print(f'==> Unexpected keys: {unexpected}')
    model = model.half()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            lr_z = pickle.loads(Path(args.src_lat).read_bytes()).to(device)
            print(f'Original Latent Shape: {lr_z.shape}')
            z_ref = upsample(lr_z, config.sr_scale)
            b, c, t, h, w = z_ref.shape
            z_ref = z_ref.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            z_ref = z_ref * config.scale_factor
            z = torch.randn(t, 4, h, w, device=device)
            if args.scheduler != 'DPMS':
                hr_z = denoise_with_diffuser_sched(args.scheduler, model, z, args.step, z_ref)
            else:
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=None, uncondition=None,
                    cfg_scale=1.0, model_kwargs=dict(mask=None))
                hr_z = dpm_solver.sample(
                    z, x_ref=z_ref, steps=args.step, order=2, skip_type='time_uniform', method='multistep',
                )
            print(f'Refined Latent Shape: {hr_z.shape}')
            hr_z = hr_z.permute(1, 0, 2, 3).unsqueeze(0) / config.scale_factor
            Path(args.dst_lat).write_bytes(pickle.dumps(hr_z.cpu()))
            print(f'Saved refined latents to {args.dst_lat}')
