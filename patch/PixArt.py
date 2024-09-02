# Modified from https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/model/nets/PixArt.py
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, T2IFinalLayer, AttentionKVCompress, TimestepEmbedder


def forward_block(block, x, t):
    if block.training:
        return auto_grad_checkpoint(block, x, t)
    return block(x, t)


class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0,
                 sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


#################################################################################
#                                 Core PixArt Model                             #
#################################################################################

@MODELS.register_module()
class PixArt(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            pred_sigma=True,
            drop_path=0.,
            pe_interpolation=1.0,
            attn_strides=None,
            lats_per_vid=-1,
            **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = 4 * 2 if pred_sigma else 4
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.hidden_size = hidden_size
        self.attn_strides = attn_strides
        self.input_size = input_size
        assert lats_per_vid in (4, 8, 16), lats_per_vid
        self.lats_per_vid = lats_per_vid

        self.x_embedder = PatchEmbed(self.patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i]) for i in range(depth)])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()


    def get_pos_embed(self, x, train_as_img):
        _, _, h, w = x.shape
        assert h == w, x.shape
        grid_size = h // self.patch_size

        if h == self.input_size:
            pe_interpolation = self.pe_interpolation
            lats_per_vid = self.lats_per_vid
        else:
            assert not train_as_img
            scale = 2
            assert h == self.input_size * scale
            pe_interpolation = self.pe_interpolation * scale
            lats_per_vid = self.lats_per_vid // (scale * scale)

        pe = get_2d_sincos_pos_embed(768, grid_size, pe_interpolation)
        n, d = pe.shape
        pe = torch.from_numpy(pe).unsqueeze(0)
        pe = torch.stack([pe]*lats_per_vid, dim=1).reshape(1,lats_per_vid*n, d)

        tgrid = np.arange(lats_per_vid, dtype=np.float32)
        te = get_1d_sincos_pos_embed_from_grid(384, tgrid)
        te = torch.from_numpy(te)
        te = torch.stack([te]*n, dim=1).reshape(lats_per_vid*n, 384).unsqueeze(0)
        embed = torch.cat([te, pe], dim=2)
        return embed.to(x.dtype).to(x.device), lats_per_vid


    def forward(self, x, timestep, train_as_img=False, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        t = self.t_embedder(timestep.to(x.dtype))
        pos_embed, lats_per_vid = self.get_pos_embed(x, train_as_img)

        x = self.x_embedder(x)
        b, n, d = x.shape

        if train_as_img:
            lats_per_vid = 1
            timestep3d = timestep
            pos_embed = pos_embed.reshape(self.lats_per_vid, n, self.hidden_size)
            pos_embed = torch.cat([pos_embed]*(b//self.lats_per_vid), dim=0)
            x3d = x
        else:
            timestep3d = timestep.reshape(-1, lats_per_vid)[:,0].contiguous()
            x3d = x.reshape(b//lats_per_vid, lats_per_vid, n, d).reshape(b//lats_per_vid, lats_per_vid*n, d)

        t3d = self.t_embedder(timestep3d)
        t3d = self.t_block(t3d)
        x3d = x3d + pos_embed
        if self.attn_strides is not None:
            attn_steps = len(self.attn_strides)
            xb, xn, xd = x3d.shape
            xh = xw = int((xn // lats_per_vid) ** 0.5)

            for bi, block in enumerate(self.blocks):
                strides = self.attn_strides[bi % attn_steps]
                if strides is None:
                    x3d = forward_block(block, x3d, t3d)
                else:
                    ts, xs = [int(v) for v in strides]
                    if train_as_img:
                        if xs == 1:
                            x3d = forward_block(block, x3d, t3d)
                        else:
                            gh, gw = xh // xs, xw // xs
                            tb, td = t3d.shape
                            t03d_grid = torch.stack([t3d]*xs*xs, dim=1).reshape(tb*xs*xs,td)
                            x3d_grid = x3d.reshape(xb, xs, gh, xs, gw, xd).permute(0, 1, 3, 2, 4, 5) #(xb, xs, xs, gh, gw, xd)
                            x3d_grid = x3d_grid.reshape(xb*xs*xs, gh*gw, xd)
                            x3d_grid = forward_block(block, x3d_grid, t03d_grid)
                            x3d_grid = x3d_grid.reshape(xb, xs, xs, gh, gw, xd).permute(0, 1, 3, 2, 4, 5) #(xb, xs, gh, xs, gw, xd)
                            x3d_grid = x3d_grid.reshape(xb, xs*gh*xs*gw, xd)
                            x3d = x3d_grid
                    else:
                        gh, gw = xh // xs, xw // xs
                        gt = lats_per_vid // ts

                        tb, td = t3d.shape
                        t03d_grid = torch.stack([t3d]*ts*xs*xs, dim=1).reshape(tb*ts*xs*xs,td)

                        x3d_grid = x3d.reshape(xb, ts, gt, xs, gh, xs, gw, xd).permute(0, 1, 3, 5, 2, 4, 6, 7) #(xb, ts, xs, xs, gt, gh, gw, xd)
                        x3d_grid = x3d_grid.reshape(xb*ts*xs*xs, gt*gh*gw, xd)
                        x3d_grid = forward_block(block, x3d_grid, t03d_grid)
                        x3d_grid = x3d_grid.reshape(xb, ts, xs, xs, gt, gh, gw, xd).permute(0, 1, 4, 2, 5, 3, 6, 7) #xb, ts, gt, xs, gh, xs, gw, xd
                        x3d_grid = x3d_grid.reshape(xb, ts*gt*xs*gh*xs*gw, xd)
                        x3d = x3d_grid
        else:
            for block in self.blocks:
                x3d = forward_block(block, x3d, t3d)

        if lats_per_vid != 1:
            x = x3d.reshape(b//lats_per_vid, lats_per_vid, n, d).reshape(b, n, d)
        else:
            x = x3d

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


    def get_ind_slices(self, size):
        assert size >= self.input_size, (size, self.input_size)
        step = round(self.input_size * 0.75)
        inds = []
        beg = 0
        while True:
            end = beg + self.input_size
            if end < size:
                inds.append(beg)
                beg += step
                continue
            beg = size - self.input_size
            inds.append(beg)
            break
        return inds


    def forward_by_slice(self, x, timestep):
        b, _, h, w = x.shape
        if h == w == self.input_size:
            return self.forward(x, timestep)
        rows = self.get_ind_slices(h)
        cols = self.get_ind_slices(w)
        y = torch.zeros((b, 8, h, w), dtype=x.dtype, device=x.device)
        for r in rows:
            for c in cols:
                y[:,:,r:r+self.input_size,c:c+self.input_size] = self.forward(x[:,:,r:r+self.input_size,c:c+self.input_size], timestep)
        return y


    def forward_with_dpmsolver(self, x, timestep, **kwargs):
        n, _, h, w = x.shape
        assert n >= self.lats_per_vid, 'No enough frames'
        z = torch.zeros((n, 8, h, w), dtype=x.dtype, device=x.device)
        t_stp = self.lats_per_vid // 2
        inds = []
        for i in range(0, n, t_stp):
            beg, end = i, i + self.lats_per_vid
            if end > n: beg, end = n - self.lats_per_vid, n
            inds.append(beg)
            if end == n: break
        pbar = tqdm(inds) if len(inds) > 1 else inds

        last_pos = 0
        for beg in pbar:
            end = beg + self.lats_per_vid
            z_slice = self.forward_by_slice(x[beg:end], timestep[beg:end])
            overlap = last_pos - beg
            if overlap < 2:
                z[beg:end] = z_slice
            else:
                weights = np.ones(self.lats_per_vid, dtype=np.float32)
                weights[:overlap] = np.linspace(0, 1, num=overlap, dtype=np.float32)
                weights = torch.from_numpy(weights).to(z.device).reshape(-1, 1, 1, 1)
                z[beg:end] = z[beg:end] * (1. - weights) + z_slice * weights
            last_pos = end
        return z.chunk(2, dim=1)[0]


    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        b, n, _ = x.shape
        h = w = int(n ** 0.5)
        x = x.reshape(b, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(b, c, h * p, w * p)
        return imgs


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


#################################################################################
#                              Position Embedding                               #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, pe_interpolation):
    if isinstance(grid_size, int): grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#                                 PixArt Configs                                #
#################################################################################

@MODELS.register_module()
def PixArt_XL_2(**kwargs):
    return PixArt(patch_size=2, depth=28, hidden_size=1152, num_heads=16, **kwargs)
