import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def encode_text(clip_model, class_names, templates, aggregate='average', return_text=False):
    all_texts = [[t.format(c.replace('_', ' ')) for t in templates] for c in class_names]
    text_features = []
    device = next(clip_model.parameters()).device
    for texts in all_texts:
        tokens = clip.tokenize(texts).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb, dim=1)
        if aggregate == 'average':
            emb = emb.mean(dim=0)
        text_features.append(emb)
    text_features = torch.stack(text_features, dim=0)
    text_features = F.normalize(text_features, dim=1)
    if return_text:
        return text_features, all_texts
    return text_features


def encode_text_single(clip_model, class_names, template):
    texts = [template.format(c.replace('_', ' ')) for c in class_names]
    device = next(clip_model.parameters()).device
    tokens = clip.tokenize(texts).to(device)
    emb = clip_model.encode_text(tokens)
    emb = F.normalize(emb, dim=1)
    return emb


def configure_model(model, freeze_text_encoder=False, freeze_image_encoder=False, mode='norm'):
    model.eval()
    model.requires_grad_(False)

    if mode == 'norm':  # adapt the normalization layers

        if not freeze_text_encoder:
            for m in model.transformer.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
            model.ln_final.eval()
            model.ln_final.weight.requires_grad_(True)
            model.ln_final.bias.requires_grad_(True)

        if not freeze_image_encoder:
            for m in model.visual.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)

    else:
        raise ValueError('Unknown mode for configure_model')

    return model


class UnimodalCLIP(nn.Module):
    def __init__(self, clip_model, text_feature):
        super().__init__()
        self.clip_model = clip_model
        self.text_feature = text_feature

    def forward(self, images):
        img_pre = self.clip_model.encode_image(images)
        img_f = F.normalize(img_pre, dim=1)
        return 100.0 * img_f @ self.text_feature.T
