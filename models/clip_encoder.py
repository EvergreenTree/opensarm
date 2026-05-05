import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import List
from PIL import Image

class FrozenCLIPEncoder(nn.Module):
    def __init__(self, ckpt: str, device: torch.device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(ckpt).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(ckpt)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list[str], length B
        returns: (B, 512) CLIP text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        if not isinstance(text_embeds, torch.Tensor):
            if hasattr(text_embeds, "text_embeds"):
                text_embeds = text_embeds.text_embeds
            elif hasattr(text_embeds, "pooler_output"):
                text_embeds = self.model.text_projection(text_embeds.pooler_output)
            else:
                raise TypeError(f"Unexpected CLIP text output type: {type(text_embeds)}")
        return text_embeds

    def encode_image(self, images: List[Image.Image], do_rescale=False) -> torch.Tensor:
        """
        images: list of PIL Images, length B
        returns: (B, 512) CLIP image embeddings
        """
        inputs = self.processor(images=images, return_tensors="pt", do_rescale=do_rescale).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        if not isinstance(image_embeds, torch.Tensor):
            if hasattr(image_embeds, "image_embeds"):
                image_embeds = image_embeds.image_embeds
            elif hasattr(image_embeds, "pooler_output"):
                image_embeds = self.model.visual_projection(image_embeds.pooler_output)
            else:
                raise TypeError(f"Unexpected CLIP image output type: {type(image_embeds)}")
        return image_embeds
