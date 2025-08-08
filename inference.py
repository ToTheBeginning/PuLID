import json
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from pulid import attention_processor as attention
from pulid.pipeline_v1_1 import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

torch.set_grad_enabled(False)


def model_fn(model_dir):
    """Loads the model when the SageMaker container starts."""
    base_model = 'RunDiffusion/Juggernaut-XL-v9'  # Change as needed
    sampler = 'dpmpp_2m'  # Default sampler
    pipeline = PuLIDPipeline(sdxl_repo=base_model, sampler=sampler)
    return pipeline


def input_fn(request_body, request_content_type):
    """Processes input from SageMaker requests."""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Load image
        id_image_bytes = BytesIO(bytes(data["id_image"]))
        id_image = Image.open(id_image_bytes).convert("RGB")
        id_image = np.array(id_image)

        supp_images = []
        for supp_img_bytes in data.get("supp_images", []):
            supp_img = Image.open(BytesIO(bytes(supp_img_bytes))).convert("RGB")
            supp_images.append(np.array(supp_img))

        return {
            "id_image": id_image,
            "supp_images": supp_images,
            "prompt": data["prompt"],
            "neg_prompt": data.get("neg_prompt", ""),
            "scale": data.get("scale", 7.0),
            "seed": data.get("seed", 2691993),
            "steps": data.get("steps", 20),
            "H": data.get("H", 1024),
            "W": data.get("W", 1024),
            "id_scale": data.get("id_scale", 1.0),
            "num_zero": data.get("num_zero", 0),
            "ortho": data.get("ortho", "none")
        }
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Runs inference."""
    id_image = resize_numpy_image_long(input_data["id_image"], 1024)
    supp_id_image_list = [
        resize_numpy_image_long(supp_img, 1024) for supp_img in input_data["supp_images"]
    ]
    id_image_list = [id_image] + supp_id_image_list

    if id_image_list:
        uncond_id_embedding, id_embedding = model.get_id_embedding(id_image_list)
    else:
        uncond_id_embedding = None
        id_embedding = None

    img = model.inference(
        input_data["prompt"],
        (1, input_data["H"], input_data["W"]),
        input_data["neg_prompt"],
        id_embedding,
        uncond_id_embedding,
        input_data["id_scale"],
        input_data["scale"],
        input_data["steps"],
        input_data["seed"]
    )[0]

    img_buffer = BytesIO()
    Image.fromarray(img).save(img_buffer, format="PNG")
    return img_buffer.getvalue()


def output_fn(prediction, response_content_type):
    """Formats the model output."""
    if response_content_type == "image/png":
        return prediction
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
