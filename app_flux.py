import time

import gradio as gr
import torch
from einops import rearrange
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long


def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model, version=args.version)

    @torch.inference_mode()
    def generate_image(
            self,
            width,
            height,
            num_steps,
            start_step,
            guidance,
            seed,
            prompt,
            id_image=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list

_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">PuLID for FLUX</h1>
    <p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='https://arxiv.org/abs/2404.16022' target='_blank'>PuLID: Pure and Lightning ID Customization via Contrastive Alignment</a> | Codes: <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'>GitHub</a></p>
</div>

❗️❗️❗️**Tips:**
- `timestep to start inserting ID:` The smaller the value, the higher the fidelity, but the lower the editability; the higher the value, the lower the fidelity, but the higher the editability. **The recommended range for this value is between 0 and 4**. For photorealistic scenes, we recommend using 4; for stylized scenes, we recommend using 0-1. If you are not satisfied with the similarity, you can lower this value; conversely, if you are not satisfied with the editability, you can increase this value.
- `true CFG scale:` In most scenarios, it is recommended to use a fake CFG, i.e., setting the true CFG scale to 1, and just adjusting the guidance scale. This is also more efficiency. However, in a few cases, utilizing a true CFG can yield better results. For more detaileds, please refer to the [doc](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md#useful-tips).
- please refer to the <a href='https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md' target='_blank'>github doc</a> for more details and info about the model, we provide the detail explanation about the above two parameters in the doc.
- we provide some examples in the bottom, you can try these example prompts first

'''  # noqa E501

_CITE_ = r"""
If PuLID is helpful, please help to ⭐ the <a href='https://github.com/ToTheBeginning/PuLID' target='_blank'> Github Repo</a>. Thanks!
---

📧 **Contact**
If you have any questions or feedbacks, feel free to open a discussion or contact <b>wuyanze123@gmail.com</b>.
"""  # noqa E501


def create_demo(args, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                offload: bool = False, aggressive_offload: bool = False):
    generator = FluxGenerator(model_name, device, offload, aggressive_offload, args)

    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="portrait, color, cinematic")
                id_image = gr.Image(label="ID Image")
                id_weight = gr.Slider(0.0, 3.0, 1, step=0.05, label="id weight")

                width = gr.Slider(256, 1536, 896, step=16, label="Width")
                height = gr.Slider(256, 1536, 1152, step=16, label="Height")
                num_steps = gr.Slider(1, 20, 20, step=1, label="Number of steps")
                start_step = gr.Slider(0, 10, 0, step=1, label="timestep to start inserting ID")
                guidance = gr.Slider(1.0, 10.0, 4, step=0.1, label="Guidance")
                seed = gr.Textbox(-1, label="Seed (-1 for random)")
                max_sequence_length = gr.Slider(128, 512, 128, step=128,
                                                label="max_sequence_length for prompt (T5), small will be faster")

                with gr.Accordion("Advanced Options (True CFG, true_cfg_scale=1 means use fake CFG, >1 means use true CFG, if using true CFG, we recommend set the guidance scale to 1)", open=False):    # noqa E501
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="bad quality, worst quality, text, signature, watermark, extra limbs")
                    true_cfg = gr.Slider(1.0, 10.0, 1, step=0.1, label="true CFG scale")
                    timestep_to_start_cfg = gr.Slider(0, 20, 1, step=1, label="timestep to start cfg", visible=args.dev)

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Textbox(label="Used Seed")
                intermediate_output = gr.Gallery(label='Output', elem_id="gallery", visible=args.dev)
                gr.Markdown(_CITE_)

        with gr.Row(), gr.Column():
                gr.Markdown("## Examples")
                example_inps = [
                    [
                        'a woman holding sign with glowing green text \"PuLID for FLUX\"',
                        'example_inputs/liuyifei.png',
                        4, 4, 2680261499100305976, 1
                    ],
                    [
                        'portrait, side view',
                        'example_inputs/liuyifei.png',
                        4, 4, 1205240166692517553, 1
                    ],
                    [
                        'white-haired woman with vr technology atmosphere, revolutionary exceptional magnum with remarkable details',  # noqa E501
                        'example_inputs/liuyifei.png',
                        4, 4, 6349424134217931066, 1
                    ],
                    [
                        'a young child is eating Icecream',
                        'example_inputs/liuyifei.png',
                        4, 4, 10606046113565776207, 1
                    ],
                    [
                        'a man is holding a sign with text \"PuLID for FLUX\", winter, snowing, top of the mountain',
                        'example_inputs/pengwei.jpg',
                        4, 4, 2410129802683836089, 1
                    ],
                    [
                        'portrait, candle light',
                        'example_inputs/pengwei.jpg',
                        4, 4, 17522759474323955700, 1
                    ],
                    [
                        'profile shot dark photo of a 25-year-old male with smoke escaping from his mouth, the backlit smoke gives the image an ephemeral quality, natural face, natural eyebrows, natural skin texture, award winning photo, highly detailed face, atmospheric lighting, film grain, monochrome',  # noqa E501
                        'example_inputs/pengwei.jpg',
                        4, 4, 17733156847328193625, 1
                    ],
                    [
                        'American Comics, 1boy',
                        'example_inputs/pengwei.jpg',
                        1, 4, 13223174453874179686, 1
                    ],
                    [
                        'portrait, pixar',
                        'example_inputs/pengwei.jpg',
                        1, 4, 9445036702517583939, 1
                    ],
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, start_step, guidance, seed, true_cfg],
                            label='fake CFG')

                example_inps = [
                    [
                        'portrait, made of ice sculpture',
                        'example_inputs/lecun.jpg',
                        1, 1, 3811899118709451814, 5
                    ],
                ]
                gr.Examples(examples=example_inps, inputs=[prompt, id_image, start_step, guidance, seed, true_cfg],
                            label='true CFG')

        generate_btn.click(
            fn=generator.generate_image,
            inputs=[width, height, num_steps, start_step, guidance, seed, prompt, id_image, id_weight, neg_prompt,
                    true_cfg, timestep_to_start_cfg, max_sequence_length],
            outputs=[output_image, seed_output, intermediate_output],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument('--version', type=str, default='v0.9.1', help='version of the model', choices=['v0.9.0', 'v0.9.1'])
    parser.add_argument("--name", type=str, default="flux-dev", choices=list('flux-dev'),
                        help="currently only support flux-dev")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, for 24G GPUs")
    parser.add_argument("--fp8", action="store_true", help="use flux-dev-fp8 model")
    parser.add_argument("--onnx_provider", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="set onnx_provider to cpu (default gpu) can help reduce RAM usage, and when combined with"
                             "fp8 option, the peak RAM is under 15GB")
    parser.add_argument("--port", type=int, default=8080, help="Port to use")
    parser.add_argument("--dev", action='store_true', help="Development mode")
    parser.add_argument("--pretrained_model", type=str, help='for development')
    args = parser.parse_args()

    if args.aggressive_offload:
        args.offload = True

    demo = create_demo(args, args.name, args.device, args.offload, args.aggressive_offload)
    demo.launch(server_name='0.0.0.0', server_port=args.port)
