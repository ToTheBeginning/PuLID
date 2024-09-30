# PuLID (NeurIPS 2024)

### :open_book: PuLID: Pure and Lightning ID Customization via Contrastive Alignment
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.16022) [![xl](https://img.shields.io/badge/🤗-HuggingFaceDemo-orange)](https://huggingface.co/spaces/yanze/PuLID) [![flux](https://img.shields.io/badge/🤗-PuLID_FLUX_demo-orange)](https://huggingface.co/spaces/yanze/PuLID-FLUX) [![Replicate](https://img.shields.io/badge/Replicate-Demo_for_PuLID-blue)](https://replicate.com/zsxkib/pulid) [![Replicate](https://img.shields.io/badge/Replicate-PuLID_FLUX-blue)](https://replicate.com/zsxkib/flux-pulid)<br>
> Zinan Guo*, Yanze Wu*✝, Zhuowei Chen, Lang Chen, Peng Zhang, Qian He <br>
> (*Equal Contribution, ✝Corresponding Author) <br>
> ByteDance Inc <br>

### :triangular_flag_on_post: Updates
* **2024.09.26**: 🎉 PuLID accepted by NeurIPS 2024
* **2024.09.12**: 💥 We're thrilled to announce the release of the **PuLID-FLUX-v0.9.0 model**. Enjoy exploring its capabilities! 😊 [Learn more about this model](docs/pulid_for_flux.md)
* **2024.05.23**: share the [preview of our upcoming v1.1 model](docs/v1.1_preview.md), please stay tuned
* **2024.05.01**: release v1 codes&models, also the [🤗HuggingFace Demo](https://huggingface.co/spaces/yanze/PuLID)
* **2024.04.25**: release arXiv paper.

### :soon: update plan
- [ ] release PuLID-FLUX-v0.9.1 model in 2024.10
- [ ] release PuLID v1.1 (for SDXL) model in 2024.10

## PuLID for FLUX
Please check the doc and demo of PuLID-FLUX [here](docs/pulid_for_flux.md).

We will actively update and maintain this repository in the near future, so please stay tuned.

### updates
- [x] Local gradio demo is ready now
- [x] Online HuggingFace demo is ready now [![flux](https://img.shields.io/badge/🤗-PuLID_FLUX_demo-orange)](https://huggingface.co/spaces/yanze/PuLID-FLUX)
- [x] We have optimized the codes to support consumer-grade GPUS, and now **PuLID-FLUX can run on a 16GB graphic card**. Check the details [here](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md#local-gradio-demo)
- [x] (Community Implementation) Online Replicate demo is ready now [![Replicate](https://replicate.com/zsxkib/flux-pulid/badge)](https://replicate.com/zsxkib/flux-pulid)
- [x] Local gradio demo supports 12GB graphic card now


Below results are generated with PuLID-FLUX.
![pulid_flux_results](https://github.com/user-attachments/assets/7eafb90a-fdd1-4ae7-bc41-8c428d568848)


## Examples
Images generated with our PuLID
![examples](https://github.com/ToTheBeginning/PuLID/assets/11482921/65610b0d-ba4f-4dc3-a74d-bd60f8f5ce37)
Applications

https://github.com/ToTheBeginning/PuLID/assets/11482921/9bdd0c8a-99e8-4eab-ab9e-39bf796cc6b8

## :wrench: Dependencies and Installation
- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0](https://pytorch.org/) if you don't need flux-dev-fp8, otherwise [PyTorch >= 2.4.1](https://pytorch.org/)
```bash
# clone PuLID repo
git clone https://github.com/ToTheBeginning/PuLID.git
cd PuLID
# create conda env
conda create --name pulid python=3.10
# activate env
conda activate pulid
# Install dependent packages
# 1. if you don't need flux-fp8, e.g., you are using xl or flux-bf16, install the following requirements.txt
pip install -r requirements.txt
# 2. if you need flux-fp8 (to put flux on consumer-grade gpu), install the following requirements_fp8.txt
pip install -r requirements_fp8.txt
```

## :zap: Quick Inference
### Local Gradio Demo
```bash
python app.py
```

### Online HuggingFace Demo
Thanks for the GPU grant from HuggingFace team, you can try PuLID HF demo in 
[https://huggingface.co/spaces/yanze/PuLID](https://huggingface.co/spaces/yanze/PuLID)

## :paperclip: Related Resources
Following are some third-party implementations of PuLID we have found in the Internet. 
We appreciate the efforts of the respective developers for making PuLID accessible to a wider audience.
If there are any PuLID based resources and applications that we have not mentioned here, please let us know, 
and we will include them in this list.

#### Online Demo
- **Colab**: https://github.com/camenduru/PuLID-jupyter provided by [camenduru](https://github.com/camenduru)
- **Replicate (PuLID)**: https://replicate.com/zsxkib/pulid provided by [zsxkib](https://github.com/zsxkib)
- **Replicate (PuLID-FLUX)**: https://replicate.com/zsxkib/flux-pulid provided by [zsxkib](https://github.com/zsxkib)

#### ComfyUI
- https://github.com/cubiq/PuLID_ComfyUI provided by [cubiq](https://github.com/cubiq), native ComfyUI implementation
- https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO provided by [ZHO](https://github.com/ZHO-ZHO-ZHO), diffusers-based implementation

#### WebUI
- https://github.com/Mikubill/sd-webui-controlnet/pull/2838 provided by [huchenlei](https://github.com/huchenlei)

## Disclaimer
This project strives to impact the domain of AI-driven image generation positively. Users are granted the freedom to 
create images using this tool, but they are expected to comply with local laws and utilize it responsibly. 
The developers do not assume any responsibility for potential misuse by users.


##  Citation
If PuLID is helpful, please help to ⭐ the repo.

If you find this project useful for your research, please consider citing our paper:
```bibtex
@article{guo2024pulid,
  title={PuLID: Pure and Lightning ID Customization via Contrastive Alignment},
  author={Guo, Zinan and Wu, Yanze and Chen, Zhuowei and Chen, Lang and He, Qian},
  journal={arXiv preprint arXiv:2404.16022},
  year={2024}
}
```

## :e-mail: Contact
If you have any comments or questions, please [open a new issue](https://github.com/ToTheBeginning/PuLID/issues/new/choose) or feel free to contact [Yanze Wu](https://tothebeginning.github.io/) and [Zinan Guo](mailto:guozinan.1@bytedance.com).