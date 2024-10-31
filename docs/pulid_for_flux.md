# PuLID for FLUX
We are happy to release the **PuLID-FLUX-v0.9.0** model, which provides a tuning-free ID customization solution for FLUX.1-dev. 

If PuLID-FLUX is helpful, please help to ⭐ this repo or recommend it to your friends 😊

## Inference
### :triangular_flag_on_post: Update
- 2024.10.31: We release the **PuLID-FLUX.v0.9.1** model. Compared to the previous version, v0.9.1 has improved the ID fidelity, with an increase of about 5 percentage points in quantitative metrics of facial similarity.

### Local Gradio Demo
You first need to follow the [dependencies-and-installation](../README.md#wrench-dependencies-and-installation) to set 
up the environment, and download the `flux1-dev.safetensors` (if you want to use bf16 rather than fp8) and `ae.safetensors` from [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main).
The PuLID-FLUX model will be automatically downloaded from [huggingface](https://huggingface.co/guozinan/PuLID/tree/main).

There are following four options to run the gradio demo:

:notes: Note: The Gradio demo defaults to using the latest version of the model. If you need to switch to an older version or a specific version, please append `--version SPECIFIC_VERSION` to the following command lines.

#### naive bf16
simply run `python app_flux.py`, the peak memory is under 45GB.

#### bf16 + offload
run `python app_flux.py --offload`, the peak memory is under 30GB.

#### fp8 + offload  (for consumer-grade GPUs)
To use fp8, you need to make sure you have installed `requirements-fp8.txt`, it includes `optimum-quanto` and higher version of PyTorch.
We use `flux-dev-fp8` checkpoint from [XLabs-AI/flux-dev-fp8](https://huggingface.co/XLabs-AI/flux-dev-fp8), it will be automatically downloaded. You can also download it manually and put it in the models folder

Run `python app_flux.py --offload --fp8 --onnx_provider cpu`, the peak memory is under 15GB, this is for GPU with 16GB memory.

For 24GB graphic memory users, you can run `python app_flux.py --offload --fp8`, the peak memory is under 17GB.

For 12GB graphic memory users, you can run `python app_flux.py --aggressive_offload --fp8 --onnx_provider cpu`, the peak memory is about 11GB. 
However, using aggressive offload (like sequential offload), the speed will be very slow due to the frequent need for memory transfers between CPU and GPU at each timestep.

Please note that, there is a difference in image quality between fp8 and bf16, with some degradation in the former. 
Specifically, the details of the face may be slightly worse, but the layout is similar. If you want the best results
of PuLID-FLUX or you have the resources, please use bf16 rather than fp8.
We have included a comparison in the table below.

|      |                                            case1                                            |                                            case2                                             |                                            case3                                            |                                           case4                                          |
|------|:-------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|
| bf16 | ![c1_bf16](https://github.com/user-attachments/assets/781b2102-d5fe-4786-b4d3-7b8df501c781) | ![c2_bf16](https://github.com/user-attachments/assets/6218a6ca-f07e-4a9a-ac63-896526ff52cf)  | ![c3_bf16](https://github.com/user-attachments/assets/3b6675e5-d26e-4799-b0f3-72e4a7f9a771) |![c4_bf16](https://github.com/user-attachments/assets/b4e162ca-da8b-4e68-8d6b-ba1a674b2a0b)|
| fp8  | ![c1_fp8](https://github.com/user-attachments/assets/8547f020-bd39-4e9b-aa82-b85be4efc41c)  |  ![c2_fp8](https://github.com/user-attachments/assets/00d3d485-0298-4966-82e1-a31946797ac8)  | ![c3_fp8](https://github.com/user-attachments/assets/b1c6a6b6-1140-49a3-93bd-1245ee5fef4c)  |![c4_fp8](https://github.com/user-attachments/assets/62e512ca-6315-4a89-9350-430e20b86b36)|


#### bf16 + more agreesive offload
run `python app_flux.py --aggressive_offload`, the peak memory is around 23GB.
But it will be very, very slow. If you have better solution to run bf16 under 24GB, please let us know.

### Online Demo
- huggingface demo: 
[https://huggingface.co/spaces/yanze/PuLID-FLUX](https://huggingface.co/spaces/yanze/PuLID-FLUX)

### ComfyUI
Please stay tuned for the community implementation

## Visual Results
![pulid_flux_results](https://github.com/user-attachments/assets/7eafb90a-fdd1-4ae7-bc41-8c428d568848)


## Useful Tips
There are two parameters that are crucial and need to be set carefully:

1. `timestep to start inserting ID`: This parameter controls the timing of ID insertion. If set to 0, the ID starts being inserted to the DIT from the first timestep. The earlier it is inserted, the higher the ID fidelity will be, but the editability may decrease. The later it is inserted, the lower the fidelity to the ID, but the editability will increase, and the disruption to the original model behavior will also be smaller. For generating realistic images, we suggest setting this to 4. If you found the ID similarity is not high enough, you could try lowering this parameter accordingly. For generating stylized images, we suggest setting it to 0-1.
![start_id](https://github.com/user-attachments/assets/3866ffab-542d-4e2f-9a0c-6877c9158d49)

2. `true CFG scale`: FLUX.1-dev is a guidance distill model. The original CFG process, which required twice the number of inference steps, is distilled into a guidance scale, thereby modulating the DIT through the guidance scale to simulate the true CFG process with half the inference steps. We will refer to this as fake CFG in the following doc. Our PuLID-FLUX model can be tested under the fake CFG settings, and the guidance scale can be set to a commonly used value, such as 4. However, the model also supports using the real CFG for inference. We compare the results of using true CFG with the fake CFG in photorealistic scenarios below.
![fake_cfg_vs_true_cfg_fidelity](https://github.com/user-attachments/assets/73b44dc8-37c7-48c8-8f55-73882731126d)
As shown in the above image, in terms of ID fidelity, using fake CFG is similar to true CFG in most cases, except that in a few cases, true CFG achieves higher ID similarity. In terms of image aesthetics and facial naturalness, fake CFG performs better. However, by carefully adjusting hyperparameters, the performance of true CFG may be further improved, we leave this to the community to explore. Therefore, we recommend using fake CFG for photorealistic scenes. If you are not satisfy about the ID fidelity, you can try switching to true CFG. Additionally, as shown below, we have found that using fake CFG in stylized scenes sometimes results in lower ID similarity and poorer style response, so if you encounter these two issues in stylized scenes, please consider switching to true CFG.
![fake_cfg_vs_true_cfg_style](https://github.com/user-attachments/assets/fb042639-64e6-4bb3-a3a4-5c138793318e)

   

## Some Technical Details
- We switch the ID encoder from an MLP structure to a Transformer structure. Interested users can refer to [source code](https://github.com/ToTheBeginning/PuLID/blob/cce7cdd65b5bf283c1a39c29f2726902a3c135ca/pulid/encoders_flux.py#L122)
- Inspired by [Flamingo](https://arxiv.org/abs/2204.14198), we insert additional cross-attention blocks every few DIT blocks to interact ID features with DIT image features
- We would like to clarify that the acceleration method (lile SDXL-Lightning) serves as an
optional acceleration trick, but it is not indispensable for training PuLID. We will update the arxiv paper with the relevant details in the near future. Please stay tuned.


## limitation
The model is currently in beta version, and we have observed that the ID fidelity may not be high for some male inputs, maybe the model requires more training. If the improved model is ready, we will release it here, so please stay tuned.

## License
As long as you use FLUX.1-dev model, you should follow the [FLUX.1-dev model license](https://github.com/black-forest-labs/flux/tree/main/model_licenses)

## contact
If you have any questions or suggestions about the model, please contact [Yanze Wu](https://tothebeginning.github.io/) or open an issue/discussion here.