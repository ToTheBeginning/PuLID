import argparse

import gradio as gr
import numpy as np
import torch

from pulid import attention_processor as attention
from pulid.pipeline_v1_1 import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base',
    type=str,
    default='RunDiffusion/Juggernaut-XL-v9',
    choices=[
        'Lykon/dreamshaper-xl-lightning',
        # 'SG161222/RealVisXL_V4.0', will add it later
        'RunDiffusion/Juggernaut-XL-v9',
    ],
)
# parser.add_argument('--sampler', type=str, default='dpmpp_2m', choices=['dpmpp_sde', 'dpmpp_2m'])
parser.add_argument('--port', type=int, default=7860)
args = parser.parse_args()

use_lightning_model = 'lightning' in args.base.lower()
# currently we only support two commonly used sampler
args.sampler = 'dpmpp_sde' if use_lightning_model else 'dpmpp_2m'
if use_lightning_model:
    default_cfg = 2.0
    default_steps = 5
else:
    default_cfg = 7.0
    default_steps = 25

pipeline = PuLIDPipeline(sdxl_repo=args.base, sampler=args.sampler)

# other params
DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

dreamshaper_example_inps = [
    ['portrait, blacklight', 'example_inputs/liuyifei.png', 42, 0.8, 10],
    ['pixel art, 1boy', 'example_inputs/lecun.jpg', 42, 0.8, 10],
    [
        'cinematic film still, close up, photo of redheaded girl near grasses, fictional landscapes, (intense sunlight:1.4), realist detail, brooding mood, ue5, detailed character expressions, light amber and red, amazing quality, wallpaper, analog film grain',
        'example_inputs/liuyifei.png',
        42,
        0.8,
        10,
    ],
    [
        'A minimalist line art depiction of an Artificial Intelligence being\'s thought process, lines and nodes forming intricate patterns.',
        'example_inputs/hinton.jpeg',
        42,
        0.8,
        10,
    ],
    [
        'instagram photo, photo of 23 y.o man in black sweater, pale skin, (smile:0.4), hard shadows',
        'example_inputs/pengwei.jpg',
        42,
        0.8,
        10,
    ],
    [
        'by Tsutomu Nihei,(strange but extremely beautiful:1.4),(masterpiece, best quality:1.4),in the style of nicola samori,The Joker,',
        'example_inputs/lecun.jpg',
        1675432759740519133,
        0.8,
        10,
    ],
]

jugger_example_inps = [
    [
        'robot,simple robot,robot with glass face,ellipse head robot,(made partially out of glass),hexagonal shapes,ferns growing inside head,butterflies on head,butterflies flying around',
        'example_inputs/hinton.jpeg',
        15022214902832471291,
        0.8,
        20,
    ],
    ['sticker art, 1girl', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    [
        '1girl, cute model, Long thick Maxi Skirt, Knit sweater, swept back hair, alluring smile, working at a clothing store, perfect eyes, highly detailed beautiful expressive eyes, detailed eyes, 35mm photograph, film, bokeh, professional, 4k, highly detailed dynamic lighting, photorealistic, 8k, raw, rich, intricate details,',
        'example_inputs/liuyifei.png',
        42,
        0.8,
        20,
    ],
    ['Chinese paper-cut, 1girl', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['Studio Ghibli, 1boy', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
    ['1man made of ice sculpture', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['portrait of green-skinned shrek, wearing lacoste purple sweater', 'example_inputs/lecun.jpg', 42, 0.8, 20],
    ['1990s Japanese anime, 1girl', 'example_inputs/liuyifei.png', 42, 0.8, 20],
    ['made of little stones, portrait', 'example_inputs/hinton.jpeg', 42, 0.8, 20],
]


@torch.inference_mode()
def run(*args):
    id_image = args[0]
    supp_images = args[1:4]
    prompt, neg_prompt, scale, seed, steps, H, W, id_scale, num_zero, ortho = args[4:]
    seed = int(seed)
    if seed == -1:
        seed = torch.Generator(device="cpu").seed()

    pipeline.debug_img_list = []

    attention.NUM_ZERO = num_zero
    if ortho == 'v2':
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif ortho == 'v1':
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        attention.ORTHO = False
        attention.ORTHO_v2 = False

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        supp_id_image_list = [
            resize_numpy_image_long(supp_id_image, 1024) for supp_id_image in supp_images if supp_id_image is not None
        ]
        id_image_list = [id_image] + supp_id_image_list
        uncond_id_embedding, id_embedding = pipeline.get_id_embedding(id_image_list)
    else:
        uncond_id_embedding = None
        id_embedding = None

    img = pipeline.inference(
        prompt, (1, H, W), neg_prompt, id_embedding, uncond_id_embedding, id_scale, scale, steps, seed
    )[0]

    return np.array(img), str(seed), pipeline.debug_img_list


_HEADER_ = '''
<h2><b>Official Gradio Demo</b></h2><h2><a href='https://github.com/ToTheBeginning/PuLID' target='_blank'><b>PuLID: Pure and Lightning ID Customization via Contrastive Alignment</b></a></h2>

**PuLID** is a tuning-free ID customization approach. PuLID maintains high ID fidelity while effectively reducing interference with the original model’s behavior.

Code: <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>GitHub</a>. Paper: <a href='https://arxiv.org/abs/2404.16022' target='_blank'>ArXiv</a>.

❗️❗️❗️**Tips:**
- we provide some examples in the bottom, you can try these example prompts first
- a single ID image is usually sufficient, you can also supplement with additional auxiliary images
- You can adjust the trade-off between ID fidelity and editability in the advanced options, but generally, the default settings are good enough.

'''  # noqa E501

_CITE_ = r"""
If PuLID is helpful, please help to ⭐ the <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/ToTheBeginning/PuLID?style=social)](https://github.com/ToTheBeginning/PuLID)
---
📧 **Contact**
If you have any questions, feel free to open a discussion or contact us at <b>wuyanze123@gmail.com</b> or <b>guozinan.1@bytedance.com</b>.
"""  # noqa E501


with gr.Blocks(title="PuLID", css=".gr-box {border-color: #8136e2}") as demo:
    gr.Markdown(_HEADER_)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                face_image = gr.Image(label="ID image (main)", height=256)
                supp_image1 = gr.Image(label="Additional ID image (auxiliary)", height=256)
                supp_image2 = gr.Image(label="Additional ID image (auxiliary)", height=256)
                supp_image3 = gr.Image(label="Additional ID image (auxiliary)", height=256)
            prompt = gr.Textbox(label="Prompt", value='portrait,color,cinematic,in garden,soft light,detailed face')
            submit = gr.Button("Generate")
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
            scale = gr.Slider(
                label="CFG (recommend 2 for lightning model and 7 for non-accelerated model)",
                value=default_cfg,
                minimum=1,
                maximum=10,
                step=0.1,
            )
            seed = gr.Textbox(-1, label="Seed (-1 for random)")
            steps = gr.Slider(label="Steps", value=default_steps, minimum=1, maximum=30, step=1)
            with gr.Row():
                H = gr.Slider(label="Height", value=1152, minimum=512, maximum=2024, step=64)
                W = gr.Slider(label="Width", value=896, minimum=512, maximum=2024, step=64)
            with gr.Row(), gr.Accordion(
                "Advanced Options (adjust the trade-off between ID fidelity and editability)", open=False
            ):
                id_scale = gr.Slider(
                    label="ID scale (Increasing it enhances ID similarity but reduces editability)",
                    minimum=0,
                    maximum=5,
                    step=0.05,
                    value=0.8,
                    interactive=True,
                )
                num_zero = gr.Slider(
                    label="num zero (Increasing it enhances ID editability but reduces similarity)",
                    minimum=0,
                    maximum=80,
                    step=1,
                    value=20,
                    interactive=True,
                )
                ortho = gr.Dropdown(label="ortho", choices=['off', 'v1', 'v2'], value='v2', visible=False)

        with gr.Column():
            output = gr.Image(label="Generated Image")
            seed_output = gr.Textbox(label="Used Seed")
            intermediate_output = gr.Gallery(label='DebugImage', elem_id="gallery", visible=False)
            gr.Markdown(_CITE_)

    with gr.Row(), gr.Column():
        gr.Markdown("## Examples")
        if args.base == 'Lykon/dreamshaper-xl-lightning':
            gr.Examples(
                examples=dreamshaper_example_inps,
                inputs=[prompt, face_image, seed, id_scale, num_zero],
                label='dreamshaper-xl-lightning examples',
            )
        elif args.base == 'RunDiffusion/Juggernaut-XL-v9':
            gr.Examples(
                examples=jugger_example_inps,
                inputs=[prompt, face_image, seed, id_scale, num_zero],
                label='Juggernaut-XL-v9 examples',
            )

    inps = [
        face_image,
        supp_image1,
        supp_image2,
        supp_image3,
        prompt,
        neg_prompt,
        scale,
        seed,
        steps,
        H,
        W,
        id_scale,
        num_zero,
        ortho,
    ]
    submit.click(fn=run, inputs=inps, outputs=[output, seed_output, intermediate_output])

demo.launch(server_name='0.0.0.0', server_port=args.port)
