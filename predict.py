from cog import BasePredictor, Input, Path
from os.path import join
import torch
import time
import subprocess
import os

from syncdiffusion.syncdiffusion_model import SyncDiffusion
from syncdiffusion.utils import seed_everything

SD_2_BASE_URL = "https://weights.replicate.delivery/default/syncdiffusion/stable-diffusion-2-base-cache.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading from: ", url)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        if not os.path.exists("stable-diffusion-2-base"):
            download_weights(SD_2_BASE_URL, "./stable-diffusion-2-base")
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.syncdiffusion_model = SyncDiffusion(device, sd_version="2.0")

    def predict(
        self,
        prompt: str = Input(
            description="Prompt to generate from",
        ),
        negative_prompt: str = Input(
            description="Prompt for negative conditioning", default=""
        ),
        width: int = Input(
            description="Width of the output image", ge=512, le=3072, default=2048
        ),
        height: int = Input(
            description="Height of the output image", ge=512, le=3072, default=512
        ),
        guidance_scale: float = Input(
            description="Scale of the guidance image", ge=0.0, le=20.0, default=7.5
        ),
        sync_weight: float = Input(
            description="Weight of the SyncDiffusion", ge=0.0, le=30.0, default=20.0
        ),
        sync_decay_rate: float = Input(
            description="SyncDiffusion weight scheduler decay rate",
            ge=0.0,
            le=1.0,
            default=0.99,
        ),
        sync_freq: int = Input(
            description="Frequency for the SyncDiffusion", default=1
        ),
        sync_threshold: int = Input(
            description="Maximum number of steps applied with SyncDiffusion",
            ge=0,
            le=50,
            default=5,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps for the diffusion",
            ge=1,
            le=200,
            default=50,
        ),
        stride: int = Input(description="Window stride for the diffusion", default=16),
        seed: int = Input(description="Seed for the SyncDiffusion", ge=0, default=2),
        loop_closure: bool = Input(description="Use loop closure", default=False),
    ) -> Path:
        seed_everything(seed)

        # Generate images
        img = self.syncdiffusion_model.sample_syncdiffusion(
            prompts=prompt,
            negative_prompts=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            sync_weight=sync_weight,
            sync_decay_rate=sync_decay_rate,
            sync_freq=sync_freq,
            sync_thres=sync_threshold,
            stride=stride,
            loop_closure=loop_closure,
        )
        final_image_path = "output.png"
        img.save(final_image_path)

        return Path(final_image_path)
