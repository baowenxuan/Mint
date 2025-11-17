import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .utils import encode_text


class CLIP(nn.Module):

    def __init__(self, clip_model, class_names, cfg):
        super(CLIP, self).__init__()

        clip_model.eval()
        self.clip_model = clip_model

        templates = [
            'a photo of a {}.'
        ]

        with torch.no_grad():
            self.clip_weights = encode_text(clip_model, class_names, templates, aggregate='average')

    @torch.no_grad()
    def forward(self, images):

        if len(images.shape) == 4:  # batch of raw images, BS x 3 x 224 x 224
            image_features = F.normalize(self.clip_model.encode_image(images), dim=1)
        else:
            raise Exception('Image must be 4D')

        logits = 100.0 * image_features @ self.clip_weights.T

        return logits.detach()

    def reset(self):
        pass  # the stage never change for raw CLIP. Don't need to reset model.
