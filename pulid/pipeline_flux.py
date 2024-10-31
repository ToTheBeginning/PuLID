import gc

import cv2
import insightface
import torch
import torch.nn as nn
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from pulid.encoders_transformer import IDFormer, PerceiverAttentionCA
from pulid.utils import img2tensor, tensor2img


class PuLIDPipeline(nn.Module):
    def __init__(self, dit, device, weight_dtype=torch.bfloat16, onnx_provider='gpu', *args, **kwargs):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        double_interval = 2
        single_interval = 4

        # init encoder
        self.pulid_encoder = IDFormer().to(self.device, self.weight_dtype)

        num_ca = 19 // double_interval + 38 // single_interval
        if 19 % double_interval != 0:
            num_ca += 1
        if 38 % single_interval != 0:
            num_ca += 1
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.weight_dtype) for _ in range(num_ca)
        ])

        dit.pulid_ca = self.pulid_ca
        dit.pulid_double_interval = double_interval
        dit.pulid_single_interval = single_interval

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.weight_dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' \
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx',
                                                            providers=providers)
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        # self.load_pretrain()

        # other configs
        self.debug_img_list = []

    def components_to_device(self, device):
        # everything but pulid_ca
        self.face_helper.face_det = self.face_helper.face_det.to(device)
        self.face_helper.face_parse = self.face_helper.face_parse.to(device)
        self.clip_vision_model = self.clip_vision_model.to(device)
        self.pulid_encoder = self.pulid_encoder.to(device)

    def load_pretrain(self, pretrain_path=None, version='v0.9.0'):
        hf_hub_download('guozinan/PuLID', f'pulid_flux_{version}.safetensors', local_dir='models')
        ckpt_path = f'models/pulid_flux_{version}.safetensors'
        if pretrain_path is not None:
            ckpt_path = pretrain_path
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get antelopev2 embedding
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                -1
            ]  # only use the maximum face
            id_ante_embedding = face_info['embedding']
            self.debug_img_list.append(
                image[
                    int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                ]
            )
        else:
            id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding
