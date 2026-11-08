"""Load bundled configurations and local weight files."""

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

from .controlnet import ControlNetModel
from .pipeline import NWMPipeline
from .unet import UNetSpatioTemporalConditionModel


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_model(model_class, config_name, checkpoint, device, dtype):
    config = json.loads((CONFIG_DIR / f"{config_name}.json").read_text())
    with init_empty_weights():
        if model_class is CLIPVisionModelWithProjection:
            model = model_class(CLIPVisionConfig(**config))
        else:
            model = model_class.from_config(config)
    if checkpoint.suffix == ".safetensors":
        state = load_file(str(checkpoint), device="cpu")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = {name: tensor.to(dtype=dtype) for name, tensor in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def load_pipeline(weights_file, device="cuda"):
    weights_file = Path(weights_file).resolve()
    weights = json.loads(weights_file.read_text())
    weights = {name: weights_file.parent / path for name, path in weights.items()}

    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    components = {
        "unet": UNetSpatioTemporalConditionModel,
        "controlnet": ControlNetModel,
        "null_model": UNetSpatioTemporalConditionModel,
        "vae": AutoencoderKLTemporalDecoder,
        "image_encoder": CLIPVisionModelWithProjection,
    }
    models = {}
    for name, model_class in components.items():
        print(f"Loading {name}...", flush=True)
        models[name] = _load_model(model_class, name, weights[name], device, dtype)

    pipeline = NWMPipeline(
        **models,
        scheduler=EulerDiscreteScheduler.from_config(
            json.loads((CONFIG_DIR / "scheduler.json").read_text())
        ),
        feature_extractor=CLIPImageProcessor.from_dict(
            json.loads((CONFIG_DIR / "feature_extractor.json").read_text())
        ),
    )
    pipeline.unet.enable_forward_chunking()
    pipeline.null_model.enable_forward_chunking()
    return pipeline
