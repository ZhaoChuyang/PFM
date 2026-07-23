import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lpips import LPIPS
from transformers import CLIPVisionModel, ConvNextV2Model, Dinov2Model, SiglipVisionModel


DINO_DEFAULT_PATH = "facebook/dinov2-base"
CLIP_DEFAULT_PATH = "openai/clip-vit-base-patch32"
CONVNEXT_DEFAULT_PATH = "facebook/convnextv2-base-22k-384"
SIGLIP2_DEFAULT_PATH = "google/siglip2-base-patch16-512"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        net = net.lower()
        self.net = net

        if net == "vgg":
            self.model = LPIPS(net="vgg", pretrained=True)
            self.image_mean = (0, 0, 0)
            self.image_std = (1, 1, 1)
            self.input_size = 224
            self.selected_layers = None
        elif net == "dino":
            self.model = Dinov2Model.from_pretrained(DINO_DEFAULT_PATH)
            self.image_mean = IMAGENET_MEAN
            self.image_std = IMAGENET_STD
            self.input_size = 518
            self.selected_layers = [4, 6, 8, 10]
        elif net == "dinov2":
            self.model = Dinov2Model.from_pretrained(DINO_DEFAULT_PATH)
            self.image_mean = IMAGENET_MEAN
            self.image_std = IMAGENET_STD
            self.input_size = 518
            self.selected_layers = None
        elif net == "clip":
            self.model = CLIPVisionModel.from_pretrained(CLIP_DEFAULT_PATH)
            self.image_mean = tuple(
                getattr(self.model.config, "image_mean", (0.48145466, 0.4578275, 0.40821073))
            )
            self.image_std = tuple(
                getattr(self.model.config, "image_std", (0.26862954, 0.26130258, 0.27577711))
            )
            self.input_size = getattr(self.model.config, "image_size", 224)
            self.selected_layers = [4, 6, 8, 10]
        elif net == "convnext":
            self.model = ConvNextV2Model.from_pretrained(CONVNEXT_DEFAULT_PATH)
            self.image_mean = IMAGENET_MEAN
            self.image_std = IMAGENET_STD
            self.input_size = 384
            self.selected_layers = [1, 2, 3]
        elif net == "siglip2":
            self.model = SiglipVisionModel.from_pretrained(SIGLIP2_DEFAULT_PATH)
            self.image_mean = tuple(
                getattr(self.model.config, "image_mean", (0.5, 0.5, 0.5))
            )
            self.image_std = tuple(
                getattr(self.model.config, "image_std", (0.5, 0.5, 0.5))
            )
            self.input_size = getattr(self.model.config, "image_size", 512)
            self.selected_layers = [4, 6, 8, 10]
        else:
            raise ValueError(f"Unsupported perceptual model: {net}")

        self.model.requires_grad_(False)
        self.model.eval()
        self.loss_weight = 1.0

    def _prepare_inputs(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = TF.normalize(x, self.image_mean, self.image_std)
        return x

    @staticmethod
    def _l2_normalize(x, eps=1e-6):
        return x / torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if self.net == "vgg":
            return self.model(inputs, targets).mean()

        elif self.net == "dino":
            inputs = self._prepare_inputs(inputs)
            targets = self._prepare_inputs(targets)

            input_outputs = self.model(pixel_values=inputs, output_hidden_states=True)
            input_feats = [input_outputs.hidden_states[i][:, 1:] for i in self.selected_layers]

            with torch.no_grad():
                target_outputs = self.model(pixel_values=targets, output_hidden_states=True)
                target_feats = [target_outputs.hidden_states[i][:, 1:] for i in self.selected_layers]

            loss = sum(
                1 - F.cosine_similarity(inp, tgt, dim=-1).mean()
                for inp, tgt in zip(input_feats, target_feats)
            ) / len(self.selected_layers)
            return loss

        elif self.net == "dinov2":
            inputs = self._prepare_inputs(inputs)
            targets = self._prepare_inputs(targets)

            out_p = self.model(pixel_values=inputs, output_hidden_states=True)
            with torch.no_grad():
                out_t = self.model(pixel_values=targets, output_hidden_states=True)

            hs_p = out_p.hidden_states
            hs_t = out_t.hidden_states
            n_layers = len(hs_p)
            losses = []
            for i in range(1, n_layers):
                fp = self._l2_normalize(hs_p[i])
                ft = self._l2_normalize(hs_t[i])
                losses.append((fp - ft).pow(2).mean(dim=(1, 2)))

            return torch.stack(losses, dim=0).mean(dim=0).mean()

        elif self.net == "clip":
            inputs = self._prepare_inputs(inputs)
            targets = self._prepare_inputs(targets)

            input_outputs = self.model(pixel_values=inputs, output_hidden_states=True)
            input_hidden_states = input_outputs.hidden_states
            input_feat_list = [input_hidden_states[idx][:, 1:] for idx in self.selected_layers]

            with torch.no_grad():
                target_outputs = self.model(pixel_values=targets, output_hidden_states=True)
                target_hidden_states = target_outputs.hidden_states
                target_feat_list = [target_hidden_states[idx][:, 1:] for idx in self.selected_layers]

            loss = sum(
                1 - F.cosine_similarity(inp, tgt, dim=-1).mean()
                for inp, tgt in zip(input_feat_list, target_feat_list)
            ) / len(self.selected_layers)
            return loss

        elif self.net == "convnext":
            inputs = self._prepare_inputs(inputs)
            targets = self._prepare_inputs(targets)

            input_outputs = self.model(pixel_values=inputs, output_hidden_states=True)
            input_hidden_states = input_outputs.hidden_states
            input_feat_list = [input_hidden_states[idx] for idx in self.selected_layers]

            with torch.no_grad():
                target_outputs = self.model(pixel_values=targets, output_hidden_states=True)
                target_hidden_states = target_outputs.hidden_states
                target_feat_list = [target_hidden_states[idx] for idx in self.selected_layers]

            loss = sum(
                F.l1_loss(inp, tgt)
                for inp, tgt in zip(input_feat_list, target_feat_list)
            ) / len(self.selected_layers)
            return loss

        elif self.net == "siglip2":
            inputs = self._prepare_inputs(inputs)
            targets = self._prepare_inputs(targets)

            input_outputs = self.model(pixel_values=inputs, output_hidden_states=True)
            input_hidden_states = input_outputs.hidden_states
            input_feat_list = [input_hidden_states[idx] for idx in self.selected_layers]

            with torch.no_grad():
                target_outputs = self.model(pixel_values=targets, output_hidden_states=True)
                target_hidden_states = target_outputs.hidden_states
                target_feat_list = [target_hidden_states[idx] for idx in self.selected_layers]

            loss = sum(
                1 - F.cosine_similarity(inp, tgt, dim=-1).mean()
                for inp, tgt in zip(input_feat_list, target_feat_list)
            ) / len(self.selected_layers)
            return loss
