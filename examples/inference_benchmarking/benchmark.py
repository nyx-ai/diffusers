from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
import argparse
from timeit import default_timer as timer
import logging
import subprocess as sp
from typing import List


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True


# Load the pipeline in full-precision and place its model components on CUDA.
def load_model(args):
    torch_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    if args.model_type == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)
    elif args.model_type == 'sd15':
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype, safety_checker=None, require_safety_checker=False)
    else:
        raise Exception(f'Unexpected model type {args.model_type}')
    pipe = pipe.to("cuda")
    # Run the attention ops without SDPA.
    # pipe.unet.set_default_attn_processor()
    # pipe.vae.set_default_attn_processor()
    return pipe


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"
    mem_info = sp.check_output(command.split()).decode().strip().split('\n')[1:]
    def _to_numeric(inp):
        return float(inp.strip().replace('MiB', '').strip())
    mem_info = [[_to_numeric(m)/1024 for m in mem.split(',')] for mem in mem_info]
    return mem_info


def log_gpu_memory(after_phase: str):
    mem = get_gpu_memory()
    mem_str = ''.join(f'[GPU{i}: {m[0]:.2f}/{m[1]:.2f}GB]' for i, m in enumerate(mem))
    logger.info(f'GPU memory after {after_phase}: {mem_str}')


def log_time_taken(time_taken: List[float], phase: str):
    time_taken = torch.tensor(time_taken)
    mean = torch.mean(time_taken).item()
    std = torch.std(time_taken).item()
    min_val = torch.min(time_taken).item()
    logger.info(f'Time taken for {phase} {mean:.2f} ± {std:.4f} s/img (min: {min_val:.2f}s)')


def run(args):
    ts_load = timer()
    pipe = load_model(args)
    logger.info(f'Time taken model load {timer()-ts_load:.2f}s')
    log_gpu_memory('model load')

    if args.torch_compile:
        logger.info('Torch compile...')
        ts_compile = timer()
        pipe.fuse_qkv_projections()
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode=args.torch_compile_mode, fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode=args.torch_compile_mode, fullgraph=True)
        logger.info(f'... torch.compile took {timer()-ts_compile:.2f}s')

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    logger.info('Running warmup run...')
    _ = pipe(prompt, num_inference_steps=4, num_images_per_prompt=args.batch_size).images

    for num_inference_steps in args.test_num_inference_steps:
        time_taken = []
        for _ in range(args.repeats):
            ts = timer()
            _ = pipe(prompt, num_inference_steps=num_inference_steps, num_images_per_prompt=args.batch_size).images
            time_taken.append((timer() - ts)/args.batch_size)
        log_time_taken(time_taken, f'{num_inference_steps} steps')
        log_gpu_memory(f'after num inference steps {num_inference_steps}')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_type', default='sdxl', choices=['sd15', 'sdxl'], help='Model to load')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--repeats', default=3, type=int, help='Repeats')
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp32'], help='Torch dtype')
    parser.add_argument('--torch_compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--torch_compile_mode', default='default', choices=['default', 'max-autotune', 'reduce-overhead'], help='Torch compile mode')
    parser.add_argument('--test_num_inference_steps', type=int, nargs='+', default=[4, 30, 50], help='What num inference steps to benchmark')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
