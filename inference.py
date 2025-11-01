from typing import List, Tuple, Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
import time
from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.osdiff import OSDiff
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def process(
    model: OSDiff,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str
) -> Tuple[List[np.ndarray], float]:
    """
    Apply OSDiff model on a list of images.
    
    Args:
        model (OSDiff): Model.
        imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        stream_path (str): Savedir of bitstream
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        bpp
    """
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    height, width = control.size(-2), control.size(-1)
    encoder_start = time.time()
    bpp, zg = model.apply_condition_compress(control, stream_path, height, width)
    encoder_time = time.time() - encoder_start
    cond = {
        "c_latent": [model.apply_condition_decompress(stream_path)], # zc
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    
    shape = (n_samples, 4, height // 8, width // 8)

    decoder_start = time.time()
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    decoder_time = time.time() - decoder_start
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    
    return preds, bpp, encoder_time, decoder_time



def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt_sd", default='./weight/v2-1_512-ema-pruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_lc", default='path to checkpoint file of lfgcm and control module', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/osdiff.yaml', type=str, help="model config path")
    
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=1, type=int)
    
    parser.add_argument("--output", type=str, default='results/')
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()

    model: OSDiff = instantiate_from_config(OmegaConf.load(args.config))
    ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
    ckpt_lc = torch.load(args.ckpt_lc, map_location="cpu")['state_dict']
    ckpt_sd.update(ckpt_lc)
    load_state_dict(model, ckpt_sd, strict=False)
    # update preprocess model
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)

    bpps = []

    print(f"sampling {args.steps} steps using {args.sampler} sampler")

    et = []
    dt = []
    for file_path in list_image_files(args.input, follow_links=True):
        img = Image.open(file_path).convert("RGB")
        x = pad(np.array(img), scale=64)
        
        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)
        
        preds, bpp, encoder_time, decoder_time = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path
        )
        et.append(encoder_time)
        dt.append(decoder_time)
        pred = preds[0]
        bpps.append(bpp)
        
        # remove padding
        pred = pred[:img.height, :img.width, :]

        Image.fromarray(pred).save(save_path)
        print(f"save to {save_path}, bpp {bpp}")


    avg_bpp = sum(bpps) / len(bpps)
    print(f'avg bpp: {avg_bpp}')
    print(f'encodertime: {sum(et)/len(et)}',f'decodertime: {sum(dt)/len(dt)}' )
        

if __name__ == "__main__":
    main()



