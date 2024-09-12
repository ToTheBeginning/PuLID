import gradio as gr
import numpy as np
import torch

from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything

torch.set_grad_enabled(False)

pipeline = PuLIDPipeline()

# other params
DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)


def run(*args):
    id_image = args[0]
    supp_images = args[1:4]
    prompt, neg_prompt, scale, n_samples, seed, steps, H, W, id_scale, mode, id_mix = args[4:]

    pipeline.debug_img_list = []
    if mode == 'fidelity':
        attention.NUM_ZERO = 8
        attention.ORTHO = False
        attention.ORTHO_v2 = True
    elif mode == 'extremely style':
        attention.NUM_ZERO = 16
        attention.ORTHO = True
        attention.ORTHO_v2 = False
    else:
        raise ValueError

    if id_image is not None:
        id_image = resize_numpy_image_long(id_image, 1024)
        id_embeddings = pipeline.get_id_embedding(id_image)
        for supp_id_image in supp_images:
            if supp_id_image is not None:
                supp_id_image = resize_numpy_image_long(supp_id_image, 1024)
                supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
                id_embeddings = torch.cat(
                    (id_embeddings, supp_id_embeddings if id_mix else supp_id_embeddings[:, :5]), dim=1
                )
    else:
        id_embeddings = None

    seed_everything(seed)
    ims = []
    for _ in range(n_samples):
        img = pipeline.inference(prompt, (1, H, W), neg_prompt, id_embeddings, id_scale, scale, steps)[0]
        ims.append(np.array(img))

    return ims, pipeline.debug_img_list


_HEADER_ = '''
<h2><b>Official Gradio Demo</b></h2><h2><a href='https://github.com/ToTheBeginning/PuLID' target='_blank'><b>PuLID: Pure and Lightning ID Customization via Contrastive Alignment</b></a></h2>

**PuLID** is a tuning-free ID customization approach. PuLID maintains high ID fidelity while effectively reducing interference with the original model’s behavior.

Code: <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>GitHub</a>. Techenical report: <a href='https://arxiv.org/abs/2404.16022' target='_blank'>ArXiv</a>.

❗️❗️❗️**Tips:**
- we provide some examples in the bottom, you can try these example prompts first
- a single ID image is usually sufficient, you can also supplement with additional auxiliary images
- We offer two modes: fidelity mode and extremely style mode. In most cases, the default fidelity mode should suffice. If you find that the generated results are not stylized enough, you can choose the extremely style mode.

'''  # noqa E501

_CITE_ = r"""
If PuLID is helpful, please help to ⭐ the <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>Github Repo</a>. Thanks! [![GitHub Stars](https://img.shields.io/github/stars/ToTheBeginning/PuLID?style=social)](https://github.com/ToTheBeginning/PuLID)
---
🚀 **Share**
If you have generated satisfying or interesting images with PuLID, please share them with us or your friends!

📝 **Citation**
If you find our work useful for your research or applications, please cite using this bibtex:
```bibtex
@article{guo2024pulid,
  title={PuLID: Pure and Lightning ID Customization via Contrastive Alignment},
  author={Guo, Zinan and Wu, Yanze and Chen, Zhuowei and Chen, Lang and He, Qian},
  journal={arXiv preprint arXiv:2404.16022},
  year={2024}
}
```

📋 **License**
Apache-2.0 LICENSE. Please refer to the [LICENSE file](placeholder) for details.

📧 **Contact**
If you have any questions, feel free to open a discussion or contact us at <b>wuyanze123@gmail.com</b> or <b>guozinan.1@bytedance.com</b>.
"""  # noqa E501


with gr.Blocks(title="PuLID", css=".gr-box {border-color: #8136e2}") as demo:
    gr.Markdown(_HEADER_)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                face_image = gr.Image(label="ID image (main)", sources="upload", type="numpy", height=256)
                supp_image1 = gr.Image(
                    label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                )
                supp_image2 = gr.Image(
                    label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                )
                supp_image3 = gr.Image(
                    label="Additional ID image (auxiliary)", sources="upload", type="numpy", height=256
                )
            prompt = gr.Textbox(label="Prompt", value='portrait,color,cinematic,in garden,soft light,detailed face')
            submit = gr.Button("Generate")
            neg_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
            scale = gr.Slider(
                label="CFG, recommend value range [1, 1.5], 1 will be faster ",
                value=1.2,
                minimum=1,
                maximum=1.5,
                step=0.1,
            )
            n_samples = gr.Slider(label="Num samples", value=4, minimum=1, maximum=8, step=1)
            seed = gr.Slider(
                label="Seed", value=42, minimum=np.iinfo(np.uint32).min, maximum=np.iinfo(np.uint32).max, step=1
            )
            steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=100, step=1)
            with gr.Row():
                H = gr.Slider(label="Height", value=1024, minimum=512, maximum=2024, step=64)
                W = gr.Slider(label="Width", value=768, minimum=512, maximum=2024, step=64)
            with gr.Row():
                id_scale = gr.Slider(label="ID scale", minimum=0, maximum=5, step=0.05, value=0.8, interactive=True)
                mode = gr.Dropdown(label="mode", choices=['fidelity', 'extremely style'], value='fidelity')
                id_mix = gr.Checkbox(
                    label="ID Mix (if you want to mix two ID image, please turn this on, otherwise, turn this off)",
                    value=False,
                )

            gr.Markdown("## Examples")
            example_inps = [
                [
                    'portrait,cinematic,wolf ears,white hair',
                    'example_inputs/liuyifei.png',
                    'fidelity',
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, mode], label='realistic')

            example_inps = [
                [
                    'portrait, impressionist painting, loose brushwork, vibrant color, light and shadow play',
                    'example_inputs/zcy.webp',
                    'fidelity',
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, mode], label='painting style')

            example_inps = [
                [
                    'portrait, flat papercut style, silhouette, clean cuts, paper, sharp edges, minimalist,color block,man',  # noqa E501
                    'example_inputs/lecun.jpg',
                    'fidelity',
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, mode], label='papercut style')

            example_inps = [
                [
                    'woman,cartoon,solo,Popmart Blind Box, Super Mario, 3d',
                    'example_inputs/rihanna.webp',
                    'fidelity',
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, mode], label='3d style')

            example_inps = [
                [
                    'portrait, the legend of zelda, anime',
                    'example_inputs/liuyifei.png',
                    'extremely style',
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, mode], label='anime style')

            example_inps = [
                [
                    'portrait, superman',
                    'example_inputs/lecun.jpg',
                    'example_inputs/lifeifei.jpg',
                    'fidelity',
                    True,
                ]
            ]
            gr.Examples(examples=example_inps, inputs=[prompt, face_image, supp_image1, mode, id_mix], label='id mix')

        with gr.Column():
            output = gr.Gallery(label='Output', elem_id="gallery")
            intermediate_output = gr.Gallery(label='DebugImage', elem_id="gallery", visible=False)
            gr.Markdown(_CITE_)

    inps = [
        face_image,
        supp_image1,
        supp_image2,
        supp_image3,
        prompt,
        neg_prompt,
        scale,
        n_samples,
        seed,
        steps,
        H,
        W,
        id_scale,
        mode,
        id_mix,
    ]
    submit.click(fn=run, inputs=inps, outputs=[output, intermediate_output])


demo.queue(max_size=3)
demo.launch(server_name='0.0.0.0')
