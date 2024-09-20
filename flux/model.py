from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

DEVICE = torch.device("cuda")

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.pulid_ca = None
        self.pulid_double_interval = 2
        self.pulid_single_interval = 4

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        id: Tensor = None,
        id_weight: float = 1.0,
        aggressive_offload: bool = False,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        ca_idx = 0
        if aggressive_offload:
            self.double_blocks = self.double_blocks.to(DEVICE)
        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            if i % self.pulid_double_interval == 0 and id is not None:
                img = img + id_weight * self.pulid_ca[ca_idx](id, img)
                ca_idx += 1
        if aggressive_offload:
            self.double_blocks.cpu()

        img = torch.cat((txt, img), 1)
        if aggressive_offload:
            # put half of the single blcoks to gpu
            for i in range(len(self.single_blocks) // 2):
                self.single_blocks[i] = self.single_blocks[i].to(DEVICE)
        for i, block in enumerate(self.single_blocks):
            if aggressive_offload and i == len(self.single_blocks)//2:
                # put first half of the single blcoks to cpu and last half to gpu
                for j in range(len(self.single_blocks) // 2):
                    self.single_blocks[j].cpu()
                for j in range(len(self.single_blocks) // 2, len(self.single_blocks)):
                    self.single_blocks[j] = self.single_blocks[j].to(DEVICE)
            x = block(img, vec=vec, pe=pe)
            real_img, txt = x[:, txt.shape[1]:, ...], x[:, :txt.shape[1], ...]

            if i % self.pulid_single_interval == 0 and id is not None:
                real_img = real_img + id_weight * self.pulid_ca[ca_idx](id, real_img)
                ca_idx += 1

            img = torch.cat((txt, real_img), 1)
        if aggressive_offload:
            self.single_blocks.cpu()
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def components_to_gpu(self):
        # everything but double_blocks, single_blocks
        self.img_in.to(DEVICE)
        self.time_in.to(DEVICE)
        self.guidance_in.to(DEVICE)
        self.vector_in.to(DEVICE)
        self.txt_in.to(DEVICE)
        self.pe_embedder.to(DEVICE)
        self.final_layer.to(DEVICE)
        if self.pulid_ca:
            self.pulid_ca.to(DEVICE)
