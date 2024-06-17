from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.models.attention import BasicTransformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class DACNConditioningEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.seg_conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.seg_blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.seg_blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.seg_blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )

        self.seg_conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.depth_conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.depth_blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.depth_blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.depth_blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )

        self.depth_conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.cross_blk = BasicTransformerBlock(
            dim=conditioning_embedding_channels,
            num_attention_heads=8,
            attention_head_dim=40,
            cross_attention_dim=320,
            only_cross_attention=True,
            positional_embeddings="sinusoidal",
            num_positional_embeddings=4096
        )

    def forward(self, conditionings):
        if isinstance(conditionings, List):
            segment = conditionings[0]
            depth = conditionings[1]
        else:
            # duplicate input if not list
            segment = conditionings
            depth = conditionings

        # Segment embedding
        seg_embedding = self.seg_conv_in(segment)
        seg_embedding = F.silu(seg_embedding)

        for block in self.seg_blocks:
            seg_embedding = block(seg_embedding)
            seg_embedding = F.silu(seg_embedding)

        seg_embedding = self.seg_conv_out(seg_embedding)

        # Depth embedding
        depth_embedding = self.depth_conv_in(depth)
        depth_embedding = F.silu(depth_embedding)

        for block in self.depth_blocks:
            depth_embedding = block(depth_embedding)
            depth_embedding = F.silu(depth_embedding)

        depth_embedding = self.depth_conv_out(depth_embedding)

        # Add batch size dim if not present
        if seg_embedding.ndim == 3:
            seg_embedding = seg_embedding.unsqueeze(0)
        if depth_embedding.ndim == 3:
            depth_embedding = depth_embedding.unsqueeze(0)

        batch_size, channels, height, width = depth_embedding.shape
        seg_embedding = seg_embedding.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        depth_embedding = depth_embedding.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        cross_embedding = self.cross_blk(
            seg_embedding, encoder_hidden_states=depth_embedding
        )
        cross_embedding = cross_embedding.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        return cross_embedding
