import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from einops import rearrange
from IPython.display import display, Audio
import textwrap

from m2d.runtime_audio import RuntimeM2D


# Create runtime instance
weight = 'm2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth'
rt = RuntimeM2D(weight_file=weight, encoder_only=False)
project_path = "../"

def attention_forward_w_attn(self, x):
    # Based on:
    # https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py#L217
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, attn

def block_forward_w_attn(self, x):
    # Based on:
    # https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py#L245
    x_att, attn = attention_forward_w_attn(self.attn, self.norm1(x))
    x = x + self.drop_path(x_att)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x, attn


def vit_m2d_forward_encoder(self, x, mask_ratio):
    # embed patches
    x = self.patch_embed(x)

    x = x + self.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    x, x_masked2, mask, ids_restore = self.random_masking(x, mask_ratio)

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    attmap = []
    for blk in self.blocks:
        x, attn = block_forward_w_attn(blk, x)
        attmap.append(attn)
    x = self.norm(x)

    return x, mask, ids_restore, torch.stack(attmap, dim=1)
class M2DViz:
    def __init__(self, runtime):
        self.device = next(iter(runtime.parameters()))[0].device
        self.runtime = runtime
        self.grid_size = self.runtime.backbone.grid_size()
        self.patch_size = self.runtime.backbone.patch_size()

    def to_spec(self, batch_audio):
        x, stats = self.runtime.to_normalized_spec(batch_audio, return_stats=True)
        self.last_norm_stats = stats
        return x
        
    def get_attention_map(self, batch_audio, start_frame=0):
        x = self.to_spec(batch_audio)

        # trim frames
        unit_frames = self.runtime.cfg.input_size[1]
        last_frame = start_frame + unit_frames
        x = x[..., start_frame:last_frame]

        _, mask, ids_restore, attmap = vit_m2d_forward_encoder(self.runtime.backbone, x, 0.0)

        return x, mask, attmap


viz = M2DViz(rt)

import torchaudio


class BaseRawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tfms=None, random_crop=False):
        self.cfg = cfg
        self.tfms = tfms
        self.random_crop = random_crop

    def __len__(self):
        raise Exception('implement me')

    def get_audio(self, index):
        raise Exception('implement me')

    def get_label(self, index):
        return None # implement me

    def __getitem__(self, index):
        wav = self.get_audio(index) # shape is expected to be (cfg.unit_samples,)

        # Trim or stuff padding
        l = len(wav)
        unit_samples = self.cfg.unit_samples
        if l > unit_samples:
            start = np.random.randint(l - unit_samples) if self.random_crop else 0
            wav = wav[start:start + unit_samples]
        elif l < unit_samples:
            wav = F.pad(wav, (0, unit_samples - l), mode='constant', value=0)
        wav = wav.to(torch.float)

        # Apply transforms
        if self.tfms is not None:
            wav = self.tfms(wav)

        # Return item
        label = self.get_label(index)
        return wav if label is None else (wav, label)


class WavDataset(BaseRawAudioDataset):
    def __init__(self, cfg, audio_files, labels, tfms=None, random_crop=False):
        super().__init__(cfg, tfms=tfms, random_crop=random_crop)
        self.files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def get_audio(self, index):
        filename = self.files[index]
        wav, sr = torchaudio.load(filename)
        assert sr == self.cfg.sample_rate, f'Convert .wav files to {self.cfg.sample_rate} Hz. {filename} has {sr} Hz.'
        return wav[0]

    def get_label(self, index):
        return None if self.labels is None else torch.tensor(self.labels[index])


def get_batch(cfg, files, seconds=6.08):
    files = list(files)
    cfg.unit_samples = int(cfg.sample_rate*seconds)
    ds = WavDataset(cfg, files, labels=None)
    return torch.stack([ds[i] for i in range(len(ds))])

print('Please download https://github.com/nttcslab/msm-mae/releases/download/v0.0.1/AudioSetWav16k_examples.zip and unzip here!')
print('Please download AudioSetWav16k_examples.zip and unzip here!')
print('Please download AudioSetWav16k_examples.zip and unzip here!')

file_def = {
    'sound_datasets/rare_sound_event/eval/audio/babycry/mixture_evaltest_babycry_000_35c7bc20a21ec8fbb7097c6fb71487b5.wav': 'beach without babycry', # 13
}
batch = get_batch(rt.cfg, file_def.keys())
captions = list(file_def.values())
len(batch), len(captions)
x, mask, attmap = viz.get_attention_map(batch)
print(x.shape, attmap.shape)

_fd, _td = rt.backbone.patch_size()
patch_batch = rearrange(x, 'b 1 (f fd) (t td) -> b (f t) fd td', fd=_fd, td=_td)
patch_batch.shape

def show_mean_attention_map_in_rows(patch_batch, captions, attn_maps,  cols=3, points=[(-1, -1)],
                        figsize=(14, 4.2), patch_shape=(5, -1),
                        layer=-1, heads=None, input_visible=True, label_visible=True, start=0, label_chs=-1):
    B, L, H, P1, P2 = attn_maps.shape
    use_cls_token = (P1 % patch_shape[0]) > 0
    token_start = 1 if use_cls_token else 0
    if use_cls_token:
       P1 = P1 - 1
    F, T = patch_shape[0], patch_shape[1] if patch_shape[1] > 0 else P1 // patch_shape[0]
    heads = np.array(list(range(H)) if heads is None else heads)
    layer = layer if layer >= 0 else L - 1


    for r, point in enumerate(points):
        rows_per_sample = 3 if r == 0 and input_visible else 2
        _figsize = [figsize[0], figsize[1] - (0 if r == 0 else 0.8)]
        fig, axes = plt.subplots(nrows=rows_per_sample, ncols=cols, figsize=_figsize)
        for c, caption in enumerate(captions[start:start+cols]):
            if c > 0:
                for dr in range(rows_per_sample):
                    axes[dr, c].get_yaxis().set_visible(False)

            if point[0] >= 0 or point[1] >= 0:
                pf, pt = point[0] if point[0] >= 0 else F // 2, point[1] if point[1] >= 0 else T // 2
                pt_patch_idx = pf * T + pt + token_start
            else:
                pt_patch_idx = 0 # CLS
                assert use_cls_token, f'Cannot show CLS token, CLS not in the model output.'

            idx = start + c
            patches = patch_batch[idx].detach().cpu().numpy()
            df, dt = patches.shape[-2:]
            amaps = attn_maps[idx, layer, heads, pt_patch_idx, token_start:].detach().cpu().numpy()
            amap = amaps.mean(axis=0) # mean head-axis
            selience = (patches - patches.min()) * amap.reshape(-1, 1, 1)
            selience_img = rearrange(selience, '(f t) df dt -> (f df) (t dt)', f=F, t=T)
            org_img = rearrange(patches, '(f t) df dt -> (f df) (t dt)', f=F, t=T)

            cur = 0
            # input
            if input_visible and r == 0:
                axes[cur, c].imshow(org_img, origin='lower', vmin=-2, vmax=2)
                axes[cur, c].get_xaxis().set_visible(False)
                cur += 1
            # selience
            axes[cur, c].imshow(selience_img, origin='lower', vmin=0., vmax=0.08)
            axes[cur, c].get_xaxis().set_visible(False)
            if pt_patch_idx > 0:
                axes[cur, c].set_ylabel(f"({pf}, {pt})", fontsize=14)
                axes[cur, c].add_patch(mpl_patches.Rectangle((pt*dt, pf*df), dt, df,
                    edgecolor = 'red',
                    fill=False,
                    lw=2))
            cur += 1
            # attention map
            amap_img = rearrange(amap, '(f t) -> f t', f=F, t=T)
            axes[cur, c].imshow(amap_img, origin='lower', vmin=0., vmax=0.025)
            axes[cur, c].set_yticks(range(F))
            axes[cur, c].set_xticks(range(T))
            if r < len(points) - 1:
                axes[cur, c].get_xaxis().set_visible(False)
            if label_visible:
                caption = caption if label_chs < 0 else textwrap.shorten(caption, label_chs, placeholder='...')
                caption = f'({idx + 1}) {caption}'
                axes[cur, c].set_xlabel(caption, fontsize=14)
        plt.tight_layout()
        png_name = str(r)+'_'+str(point)+'_'+'spectrum.png'
        fig.savefig(project_path+'graph/no_noise/'+png_name_line)


show_mean_attention_map_in_rows(patch_batch, captions, attmap, cols=2, start=0, points=[(2, 7), (1, 12)])
show_mean_attention_map_in_rows(patch_batch, captions, attmap, cols=2, start=2, points=[(2, 7), (1, 20)])
show_mean_attention_map_in_rows(patch_batch, captions, attmap, cols=2, start=4, points=[(2, 5), (1, 15)])