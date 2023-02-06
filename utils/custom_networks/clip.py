from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import clip

class ResNetCLIPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        is_habitat: bool = True,
        pooling='attnpool',
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()

        self.is_habitat = is_habitat

        class castTensor():
            def __init__(self, type):
                self.type = type

            def __call__(self, tensor):
                return tensor.type(torch.float)

        self.rgb = "rgb" in observation_space.spaces
        self.depth = "depth" in observation_space.spaces

        if not self.is_blind:
            model, preprocess = clip.load("RN50", device=device)
            self.toPil = T.ToPILImage()

            # expected input: C x H x W (np.uint8 in [0-255])
            self.preprocess = T.Compose([
                # resize and center crop to 224
                preprocess.transforms[0],
                preprocess.transforms[1],
                # already tensor, but want float
                # T.ConvertImageDtype(torch.float),
                T.ToTensor(),
                castTensor(torch.float),
                # normalize with CLIP mean, std
                preprocess.transforms[4],
            ])
            # expected output: C x H x W (np.float32)

            self.backbone = model.visual

            if self.rgb and self.depth:
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048,)
            elif pooling == 'none':
                self.backbone.attnpool = nn.Identity()
                self.output_shape = (2048, 7, 7)
            elif pooling == 'avgpool':
                self.backbone.attnpool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(1,1)),
                    nn.Flatten()
                )
                self.output_shape = (2048,)
            else:
                self.output_shape = (1024,)

            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self.backbone.eval()

    @property
    def is_blind(self):
        return self.rgb is False and self.depth is False

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self.rgb:
            if self.is_habitat:
                rgb_observations = np.array(observations['rgb'])
            else:
                rgb_observations = np.array(observations)
            # rgb_observations = rgb_observations.transpose((2, 0, 1))  # BATCH x CHANNEL x HEIGHT X WIDTH
            rgb_observations = [self.toPil(rgb_observations)]
            rgb_observations = torch.stack(
                [self.preprocess(rgb_image) for rgb_image in rgb_observations]
            )  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32
            # rgb_observations = torch.stack(rgb_observations)
            if torch.cuda.is_available():
                rgb_observations = rgb_observations.cuda()
            rgb_x = self.backbone(rgb_observations).float()
            cnn_input.append(rgb_x)

        if self.depth:
            depth_observations = np.array([observations["depth"][..., 0]])  # [BATCH x HEIGHT X WIDTH]
            ddd = torch.stack([depth_observations] * 3, dim=1)  # [BATCH x 3 x HEIGHT X WIDTH]
            ddd = torch.stack([
                self.preprocess(TF.convert_image_dtype(depth_map, torch.uint8))
                for depth_map in ddd
            ])  # [BATCH x CHANNEL x HEIGHT X WIDTH] in torch.float32

            depth_x = self.backbone(ddd).float()
            cnn_input.append(depth_x)

        if self.rgb and self.depth:
            x = F.adaptive_avg_pool2d(cnn_input[0] + cnn_input[1], 1)
            x = x.flatten(1)
        else:
            x = torch.cat(cnn_input, dim=1)

        # TODO [CARLOS]: Check that sending tensor to CPU does not decrease performance
        return x.cpu().numpy()

