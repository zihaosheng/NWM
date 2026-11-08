# Neutral World Model

Generate a 25-frame video at 512 × 320 and 6 FPS from an initial image, an ego trajectory, and bounding boxes for every frame.

## Weights

Download the following checkpoints into `weights/`. The NWM UNet and ControlNet are our trained weights, hosted at [zihaosheng/NWM](https://huggingface.co/zihaosheng/NWM).

| Component | Source | Local path |
| --- | --- | --- |
| `unet` | [NWM UNet](https://huggingface.co/zihaosheng/NWM/blob/main/unet/diffusion_pytorch_model.safetensors) | `weights/unet/diffusion_pytorch_model.safetensors` |
| `controlnet` | [NWM ControlNet](https://huggingface.co/zihaosheng/NWM/blob/main/controlnet/diffusion_pytorch_model.bin) | `weights/controlnet/diffusion_pytorch_model.bin` |
| `null_model` | [Stability AI SVD-XT UNet](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/unet/diffusion_pytorch_model.safetensors) | `weights/null_model/diffusion_pytorch_model.safetensors` |
| `vae` | [Stability AI SVD-XT VAE](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/vae/diffusion_pytorch_model.safetensors) | `weights/vae/diffusion_pytorch_model.safetensors` |
| `image_encoder` | [Stability AI SVD-XT image encoder](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/image_encoder/model.safetensors) | `weights/image_encoder/model.safetensors` |

Inference loads these local files through the relative paths in `configs/weights.json`. Model configurations are included in `configs/`.

## Run

```bash
conda create -n nwm python=3.10 -y
conda activate nwm
pip install -r requirement.txt
```

The example uses nuScenes `scene-0416`. All 25 bounding-box lists are empty for this clip. Output is saved to `output.mp4`.

```bash
python inference.py --sample examples/nuscenes/sample.json --seed 42 --output output.mp4
```

## Input

`examples/nuscenes/sample.json` contains:

- `image`: initial RGB image path, relative to the JSON file.
- `ego_trajectory`: 25 `[right, forward]` positions in meters, relative to the initial position and aligned with the initial direction of travel. Positions at indices 5, 10, 15, and 20 form the eight-dimensional model input.
- `bounding_boxes`: 25 lists containing all visible annotated road users, ordered from farthest to nearest. Each entry has a `track_id`, `class_id`, and normalized `[x1, y1, x2, y2]` coordinates.

