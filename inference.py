import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

_MODEL_ID = "Salesforce/blip-image-captioning-base"

class Captioner:
    def __init__(self, model_id: str = _MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor & model
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)

        # generation defaults
        self.gen_kwargs = {
            "max_length": 30,
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
        }

    @torch.inference_mode()
    def caption(self, image: Image.Image) -> str:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, **self.gen_kwargs)
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
