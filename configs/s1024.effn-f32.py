_base_ = ['../PixArt_xl2_internal.py']
video_list = ['example/train.demo.txt']
data = dict(type='InternalDataSigma', transform='default_train')
sample_size = 1024
image_size = 1024
time_stride = 4
num_frames_per_video = 8
sr_scale = 4
encoder = 'evae'
decoder = 'tvae'

# model setting
model = 'PixArt_XL_2'
mixed_precision = 'fp16'
fp32_attention = True
load_from = 'pretrained/s1024.effn-f32.pth'
multi_scale = False
pe_interpolation = 2.0
sample_posterior = True
attn_strides = [
    [1, 4], [4, 1], [1, 4], [4, 1],
    [1, 4], [4, 1], [1, 4], [4, 1],
    None, None, None, None,
    [1, 4], [4, 1], [1, 4], [4, 1],
    [1, 4], [4, 1], [1, 4], [4, 1],
    None, None, None, None,
    None, None, None, None,
]
joint_training = 0.1

# training setting
num_workers = 4
train_batch_size = 4
num_epochs = 1000
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 100
log_interval = 5
save_model_epochs = 1
save_model_steps = 1000

# pixart-sigma
scale_factor = 0.13025
