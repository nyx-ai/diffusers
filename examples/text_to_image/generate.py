from diffusers import StableDiffusionPipeline
import os
from transformers import set_seed


ckpt_dir = 'out6'
out_dir = f'out_images/{ckpt_dir}'
num_samples = 20
chunk_size = 10

os.makedirs(out_dir, exist_ok=True)

set_seed(500)
pipe = StableDiffusionPipeline.from_pretrained(ckpt_dir, safety_checker=None)
pipe.to('cuda')
image_idx = 0

for idx in range(num_samples // chunk_size):
    out = pipe('photo in ukj style', num_inference_steps=20, width=512, height=512, num_images_per_prompt=chunk_size)
    for idx, img in enumerate(out.images):
        f_out = os.path.join(out_dir, f'{image_idx}.png')
        print(f'Writing to {f_out}...')
        img.save(f_out)
        image_idx += 1
