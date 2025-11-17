import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip

from .utils import encode_text, configure_model
from .TTABase import TTABase

from collections import defaultdict


class GradientAccumulator:
    def __init__(self, dim, dtype=torch.float32, device="cpu"):
        self.n = 0
        self.avg = torch.zeros(dim, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, grad):
        self.avg = (self.n / (self.n + 1)) * self.avg + (1 / (self.n + 1)) * grad
        self.n += 1

    def get(self):
        return self.avg

    def clear(self):
        self.n = 0
        self.avg.zero_()


class MeanAccumulator:
    def __init__(self, num_classes, dim, dtype=torch.float32, device="cpu"):
        self.class_sums = torch.zeros(num_classes, dim, dtype=dtype, device=device)
        self.total_sum = torch.zeros(dim, dtype=dtype, device=device)
        self.class_n = torch.zeros(num_classes, dtype=dtype, device=device)
        self.total_n = 0

        self.dtype = dtype
        self.device = device

    def update(self, embeds, preds):
        embeds, preds = embeds.detach(), preds.detach()
        self.class_sums.index_add_(dim=0, index=preds, source=embeds)
        self.total_sum += embeds.sum(dim=0)
        self.class_n.index_add_(dim=0, index=preds,
                                source=torch.ones(preds.size(0), dtype=self.dtype, device=self.device))
        self.total_n += preds.size(0)

    def get_class_means(self):
        class_means = self.class_sums / self.class_n.unsqueeze(1)
        class_means[self.class_n == 0] = 0
        return class_means

    def get_total_mean(self):
        return self.total_sum / self.total_n

    def clear(self):
        self.class_sums.zero_()
        self.total_sum.zero_()
        self.class_n.zero_()
        self.total_n = 0


class Mint(TTABase):

    def __init__(self, clip_model, class_names, cfg):
        super(Mint, self).__init__()

        clip_model.eval()
        self.clip_model = clip_model

        configure_model(self.clip_model, freeze_text_encoder=True, freeze_image_encoder=False, mode='norm')

        with torch.no_grad():
            templates = [
                "itap of a {}.",
                "a bad photo of the {}.",
                "a origami {}.",
                "a photo of the large {}.",
                "a {} in a video game.",
                "art of the {}.",
                "a photo of the small {}.",
            ]
            self.clip_weights = encode_text(clip_model, class_names, templates)

        self.device = self.clip_weights.device

        self.num_classes, self.feat_dim = self.clip_weights.shape
        self.grad_dim = sum(param.numel() for param in self.clip_model.parameters() if param.requires_grad)

        self.cfg = cfg
        self.optimizer = optim.Adam(self.clip_model.parameters(), lr=self.cfg['lr'])

        self.copy_model_and_optimizer()

        self.pre_feat_accumulator = MeanAccumulator(num_classes=self.num_classes, dim=self.feat_dim,
                                                    dtype=torch.float32, device=self.device)

        self.post_feat_accumulator = MeanAccumulator(num_classes=self.num_classes, dim=self.feat_dim,
                                                     dtype=torch.float32, device=self.device)

        self.grad_accumulator = GradientAccumulator(dim=self.grad_dim, dtype=torch.float32, device=self.device)

    def forward(self, images):

        ######## ######## ######## ######## ######## ######## ######## ########
        # Adaptation
        ######## ######## ######## ######## ######## ######## ######## ########

        with torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=1)
            logits = 100.0 * image_features @ self.clip_weights.T
            preds = torch.argmax(logits, dim=1)

            # Update embedding mean accumulator
            self.pre_feat_accumulator.update(image_features, preds)

            # Calculate PL-inter variance within this batch
            present_classes = torch.unique(preds)
            C = len(present_classes)

            class_means = self.pre_feat_accumulator.get_class_means()
            total_mean = self.pre_feat_accumulator.get_total_mean()

            counts = torch.bincount(preds, minlength=self.num_classes)
            class_weights = 1.0 / C / counts

            sample_weights = class_weights[preds]

            diff_intra = ((image_features - class_means[preds]) ** 2).sum(dim=1)
            intra_var = (diff_intra * sample_weights).sum()

            diff_total = ((image_features - total_mean) ** 2).sum(dim=1)
            total_var = (diff_total * sample_weights).sum()

            inter_var = total_var - intra_var
            loss = - inter_var  # to use the built-in optimizer

        loss.backward()

        with torch.no_grad():

            # Get gradient
            grads = [param.grad.view(-1) for param in self.clip_model.parameters() if param.requires_grad]
            flat_grad = torch.cat(grads)

            # Update grad accumulator and get aggregated gradient
            self.grad_accumulator.update(flat_grad)
            agg_grad = self.grad_accumulator.get()

            # Set gradient
            offset = 0
            for param in self.clip_model.parameters():
                if param.requires_grad:
                    numel = param.numel()
                    param.grad.copy_(agg_grad[offset:offset + numel].view_as(param))
                    offset += numel

        self.optimizer.step()
        self.optimizer.zero_grad()

        ######## ######## ######## ######## ######## ######## ######## ########
        # Inference
        ######## ######## ######## ######## ######## ######## ######## ########

        with torch.cuda.amp.autocast():
            with torch.no_grad():

                image_features = self.clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=1)
                logits = 100.0 * image_features @ self.clip_weights.T
                preds = torch.argmax(logits, dim=1)

                # Update embedding mean accumulator
                self.post_feat_accumulator.update(image_features, preds)

                class_means = self.post_feat_accumulator.get_class_means()

                n = self.post_feat_accumulator.total_n
                ratio = n / (self.cfg['prior'] + n)

                mixed_weights = (1 - ratio) * self.clip_weights + ratio * class_means
                mixed_weights = F.normalize(mixed_weights, dim=1)

                logits = 100.0 * image_features @ mixed_weights.T

        self.load_model_and_optimizer()

        return logits

    def reset(self):
        self.load_model_and_optimizer()
        self.pre_feat_accumulator.clear()
        self.post_feat_accumulator.clear()
        self.grad_accumulator.clear()
