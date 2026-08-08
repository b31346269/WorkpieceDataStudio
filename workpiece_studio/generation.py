from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


LOCAL_HF_HOME = Path(__file__).resolve().parent.parent / "workspace" / ".huggingface"
LOCAL_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(LOCAL_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_HF_HOME / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


DEFAULT_NEGATIVE_PROMPT = (
    "illustration, CGI, 3d render, toy, miniature, impossible mechanism, "
    "melted metal, warped metal, fused screws, floating bolt, duplicated screw, "
    "missing screw, extra screw, fake hole, painted circle, oval hole, collapsed hole, "
    "deformed threads, malformed fastener, asymmetrical screw head, blurry, "
    "low resolution, watermark, text"
)

STRICT_MECHANICAL_PROMPT = (
    " Preserve the reference workpiece's exact mechanical topology: keep the same "
    "number, position, diameter, circularity and depth of holes; keep every screw "
    "and bolt rigid, circular, correctly seated and aligned; preserve sharp machined "
    "edges and physically continuous metal. Vary only lighting, surface finish, "
    "camera exposure and workbench background."
)

SHAPE_VARIATION_PROMPT = (
    "Keep holes circular and drilled; keep screws rigid, aligned and seated. "
    "Redesign this as a different mechanically plausible industrial housing. "
    "Change the overall silhouette, proportions, mounting ears, ribs, cooling fins "
    "and cavity layout. Use continuous machined metal without fused parts."
)

FLUX_STRICT_PROMPT = (
    "Edit the reference photograph without changing the workpiece geometry. "
    "Keep the exact silhouette, proportions, holes, screws, ribs and mounting "
    "points. Change only material finish, lighting and background."
)

FLUX_BALANCED_PROMPT = (
    "Create a mechanically plausible variant of the reference workpiece. "
    "Moderately change its proportions, ribs and mounting details while keeping "
    "all holes circular and all fasteners rigid and correctly seated."
)

FLUX_CREATIVE_PROMPT = (
    "Use a ceiling-mounted factory inspection camera directly above the conveyor. "
    "Keep the optical axis close to perpendicular to the work surface at an 80 to "
    "90 degree elevation. The top face must appear nearly rectangular with almost "
    "parallel edges and must dominate the image. A shallow side rim may provide "
    "limited depth, but it must not dominate; do not use a three-quarter view. "
    "Completely redesign the reference as a different but functional industrial "
    "mechanical housing. Use a new silhouette, proportions, ribs, cooling fins, "
    "mounting ears and cavity layout. Keep continuous metal, circular drilled "
    "holes and rigid correctly seated fasteners."
)

FACTORY_SCENE_PROMPTS = {
    "assembly_line": (
        "Place the workpiece on a real factory assembly-line metal bench with "
        "fixtures, safety markings and overhead fluorescent lighting."
    ),
    "machine_enclosure": (
        "Place the workpiece inside a CNC machine enclosure with brushed metal "
        "walls, cool LED task lighting, mild oil residue and realistic shadows."
    ),
    "maintenance_bench": (
        "Place the workpiece on a used industrial maintenance bench with subtle "
        "oil stains, nearby hand tools and mixed warm and cool workshop light."
    ),
    "conveyor_fixture": (
        "Place the workpiece in a believable conveyor inspection fixture with "
        "clamps, machined rails and directional factory lighting."
    ),
    "warehouse_inspection": (
        "Place the workpiece at a warehouse quality-inspection station with a "
        "neutral metal surface, side daylight and overhead industrial lamps."
    ),
}


def resolve_scene_prompt(scene_preset: str, seed: int) -> tuple[str, str]:
    if scene_preset == "custom":
        return "", "custom"
    if scene_preset == "factory_mixed":
        names = tuple(FACTORY_SCENE_PROMPTS)
        scene_preset = names[seed % len(names)]
    return FACTORY_SCENE_PROMPTS.get(scene_preset, ""), scene_preset


def compose_prompt(settings: "GenerationSettings", prompt: str) -> str:
    scene_prompt, _ = resolve_scene_prompt(settings.scene_preset, settings.seed)
    return f"{scene_prompt} {prompt}".strip()


def letterbox_square(image: Image.Image, size: int = 640) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(size / source.width, size / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    background = source.resize((size, size), Image.Resampling.BILINEAR)
    background = background.filter(ImageFilter.GaussianBlur(radius=22))
    background = ImageEnhance.Brightness(background).enhance(0.55)
    x = (size - resized.width) // 2
    y = (size - resized.height) // 2
    background.paste(resized, (x, y))
    return background


def focus_crop_square(image: Image.Image, size: int = 640) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(
        source,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.47),
    )


def prepare_reference(
    image: Image.Image,
    framing: str,
    size: int = 640,
) -> Image.Image:
    if framing == "letterbox":
        return letterbox_square(image, size)
    return focus_crop_square(image, size)


@dataclass
class GenerationSettings:
    prompt: str
    negative_prompt: str
    strength: float
    ip_adapter_scale: float
    guidance_scale: float
    steps: int
    seed: int
    framing: str = "focus_crop"
    quality_mode: str = "strict"
    scene_preset: str = "factory_mixed"


class MockGenerator:
    """Workflow test generator. Its output is visibly marked and not training data."""

    def generate(self, reference: Image.Image, settings: GenerationSettings) -> Image.Image:
        rng = random.Random(settings.seed)
        image = prepare_reference(reference, settings.framing)
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.12))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.12))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, 640, 44), fill=(120, 0, 0, 210))
        draw.text((12, 12), "MOCK PREVIEW - DO NOT TRAIN", fill=(255, 255, 255, 255))
        return image


class DiffusersGenerator:
    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    ADAPTER_ID = "h94/IP-Adapter"
    ADAPTER_SUBFOLDER = "models"
    ADAPTER_WEIGHT = "ip-adapter_sd15.bin"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self._lock = RLock()
        self.runtime: dict[str, Any] = {}

    @staticmethod
    def _cached_snapshot(repo_id: str) -> str | None:
        repository = LOCAL_HF_HOME / "hub" / f"models--{repo_id.replace('/', '--')}"
        references = repository / "refs" / "main"
        snapshots = repository / "snapshots"
        if references.exists():
            revision = references.read_text(encoding="utf-8").strip()
            candidate = snapshots / revision
            if candidate.is_dir():
                return str(candidate)
        if snapshots.is_dir():
            candidates = sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
        return None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise RuntimeError(
                "Real generation dependencies are missing. Run setup.ps1 -WithML."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU was not detected. Mock mode is still available.")

        dtype = torch.float16
        model_source = self._cached_snapshot(self.MODEL_ID) or self.MODEL_ID
        adapter_source = self._cached_snapshot(self.ADAPTER_ID) or self.ADAPTER_ID
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_source,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=model_source != self.MODEL_ID,
        )
        pipe.load_ip_adapter(
            adapter_source,
            subfolder=self.ADAPTER_SUBFOLDER,
            weight_name=self.ADAPTER_WEIGHT,
            local_files_only=adapter_source != self.ADAPTER_ID,
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 10:
            pipe.enable_model_cpu_offload()
            offload = "model_cpu_offload"
        else:
            pipe.to("cuda")
            offload = "none"

        self._pipe = pipe
        self._torch = torch
        self.runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(total_vram_gb, 2),
            "offload": offload,
            "model": self.MODEL_ID,
            "adapter": f"{self.ADAPTER_ID}/{self.ADAPTER_WEIGHT}",
        }

    def generate(self, reference: Image.Image, settings: GenerationSettings) -> Image.Image:
        with self._lock:
            self._load()
            assert self._pipe is not None
            assert self._torch is not None
            strength = settings.strength
            adapter_scale = settings.ip_adapter_scale
            steps = settings.steps
            prompt = compose_prompt(settings, settings.prompt.strip())
            if settings.quality_mode == "strict":
                strength = min(strength, 0.38)
                adapter_scale = max(adapter_scale, 0.82)
                steps = max(steps, 32)
                prompt += STRICT_MECHANICAL_PROMPT
            elif settings.quality_mode == "shape_variation":
                strength = max(0.50, min(strength, 0.68))
                adapter_scale = max(0.58, min(adapter_scale, 0.76))
                steps = max(steps, 36)
                prompt = f"{SHAPE_VARIATION_PROMPT} {prompt}"
            elif settings.quality_mode == "balanced":
                strength = min(strength, 0.50)
                adapter_scale = max(adapter_scale, 0.68)
            self._pipe.set_ip_adapter_scale(adapter_scale)
            prepared = prepare_reference(reference, settings.framing)
            custom_negative = settings.negative_prompt.strip()
            negative = (
                f"{DEFAULT_NEGATIVE_PROMPT}, {custom_negative}"
                if custom_negative
                else DEFAULT_NEGATIVE_PROMPT
            )
            generator = self._torch.Generator(device="cpu").manual_seed(settings.seed)
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=prepared,
                ip_adapter_image=prepared,
                strength=strength,
                guidance_scale=settings.guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                width=640,
                height=640,
            ).images[0]
            return result.convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)


class SDXLControlNetGenerator:
    """Larger server generator with edge control for mechanical structures."""

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_ID = "diffusers/controlnet-canny-sdxl-1.0"
    ADAPTER_ID = "h94/IP-Adapter"
    ADAPTER_SUBFOLDER = "sdxl_models"
    ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self._cv2: Any | None = None
        self._lock = RLock()
        self.runtime: dict[str, Any] = {}

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import cv2
            import torch
            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetImg2ImgPipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SDXL server dependencies are missing. "
                "Run school_server/bootstrap.sh on the school server."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("SDXL + ControlNet requires a CUDA GPU.")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 20:
            raise RuntimeError(
                f"SDXL + ControlNet needs at least 20 GB VRAM; detected "
                f"{total_vram_gb:.1f} GB. Use the school RTX A5000 through SSH."
            )

        dtype = torch.float16
        model_source = DiffusersGenerator._cached_snapshot(self.MODEL_ID) or self.MODEL_ID
        control_source = (
            DiffusersGenerator._cached_snapshot(self.CONTROLNET_ID)
            or self.CONTROLNET_ID
        )
        adapter_source = (
            DiffusersGenerator._cached_snapshot(self.ADAPTER_ID) or self.ADAPTER_ID
        )
        controlnet = ControlNetModel.from_pretrained(
            control_source,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            local_files_only=control_source != self.CONTROLNET_ID,
        )
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            model_source,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            local_files_only=model_source != self.MODEL_ID,
        )
        pipe.load_ip_adapter(
            adapter_source,
            subfolder=self.ADAPTER_SUBFOLDER,
            weight_name=self.ADAPTER_WEIGHT,
            local_files_only=adapter_source != self.ADAPTER_ID,
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.to("cuda")

        self._pipe = pipe
        self._torch = torch
        self._cv2 = cv2
        self.runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(total_vram_gb, 2),
            "offload": "none",
            "model": self.MODEL_ID,
            "controlnet": self.CONTROLNET_ID,
            "adapter": f"{self.ADAPTER_ID}/{self.ADAPTER_WEIGHT}",
            "output_size": 1024,
        }

    def generate(
        self,
        reference: Image.Image,
        settings: GenerationSettings,
    ) -> tuple[Image.Image, dict[str, float | int]]:
        with self._lock:
            self._load()
            assert self._pipe is not None
            assert self._torch is not None
            assert self._cv2 is not None

            strength = settings.strength
            adapter_scale = settings.ip_adapter_scale
            steps = settings.steps
            control_scale = 0.58
            control_guidance_end = 0.92
            prompt = compose_prompt(settings, settings.prompt.strip())
            if settings.quality_mode == "strict":
                strength = min(strength, 0.32)
                adapter_scale = max(adapter_scale, 0.78)
                steps = max(steps, 36)
                control_scale = 0.90
                prompt += STRICT_MECHANICAL_PROMPT
            elif settings.quality_mode == "shape_variation":
                strength = max(0.52, min(strength, 0.68))
                adapter_scale = max(0.58, min(adapter_scale, 0.72))
                steps = max(steps, 40)
                control_scale = 0.38
                control_guidance_end = 0.48
                prompt = f"{SHAPE_VARIATION_PROMPT} {prompt}"
            elif settings.quality_mode == "balanced":
                strength = min(strength, 0.45)
                adapter_scale = max(adapter_scale, 0.68)
                steps = max(steps, 32)
                control_scale = 0.75

            prepared = prepare_reference(reference, settings.framing, size=1024)
            array = self._cv2.cvtColor(
                np.array(prepared),
                self._cv2.COLOR_RGB2GRAY,
            )
            edges = self._cv2.Canny(array, 80, 180)
            control_image = Image.fromarray(edges).convert("RGB")
            custom_negative = settings.negative_prompt.strip()
            negative = (
                f"{DEFAULT_NEGATIVE_PROMPT}, {custom_negative}"
                if custom_negative
                else DEFAULT_NEGATIVE_PROMPT
            )
            self._pipe.set_ip_adapter_scale(adapter_scale)
            generator = self._torch.Generator(device="cuda").manual_seed(settings.seed)
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=prepared,
                control_image=control_image,
                ip_adapter_image=prepared,
                strength=strength,
                controlnet_conditioning_scale=control_scale,
                control_guidance_start=0.0,
                control_guidance_end=control_guidance_end,
                guidance_scale=settings.guidance_scale,
                num_inference_steps=steps,
                generator=generator,
                width=1024,
                height=1024,
            ).images[0]
            return result.convert("RGB"), {
                "effective_strength": strength,
                "effective_ip_adapter_scale": adapter_scale,
                "effective_steps": steps,
                "controlnet_conditioning_scale": control_scale,
                "control_guidance_end": control_guidance_end,
            }


class Flux2KleinGenerator:
    """Large semantic image editor for substantial mechanical redesigns."""

    MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self._lock = RLock()
        self.runtime: dict[str, Any] = {}

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.2 dependencies are missing. Run school_server/bootstrap.sh."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("FLUX.2 Klein requires a CUDA GPU.")
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 20:
            raise RuntimeError(
                f"FLUX.2 Klein needs a 20+ GB server GPU with CPU offload; "
                f"detected {total_vram_gb:.1f} GB."
            )

        model_source = (
            DiffusersGenerator._cached_snapshot(self.MODEL_ID) or self.MODEL_ID
        )
        if model_source == self.MODEL_ID:
            raise RuntimeError(
                "FLUX.2 Klein is not fully cached. "
                "Run school_server/prefetch_flux2.py first."
            )
        pipe = Flux2KleinPipeline.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()

        self._pipe = pipe
        self._torch = torch
        self.runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(total_vram_gb, 2),
            "offload": "model_cpu_offload",
            "model": self.MODEL_ID,
            "parameters": "9B",
            "precision": "BF16",
            "output_size": 1024,
            "license": "FLUX Non-Commercial License",
            "manual_review_required": True,
        }

    def generate(
        self,
        reference: Image.Image,
        settings: GenerationSettings,
    ) -> tuple[Image.Image, dict[str, float | int | str]]:
        with self._lock:
            self._load()
            assert self._pipe is not None
            assert self._torch is not None

            mode_prompt = {
                "strict": FLUX_STRICT_PROMPT,
                "shape_variation": SHAPE_VARIATION_PROMPT,
                "balanced": FLUX_BALANCED_PROMPT,
                "creative": FLUX_CREATIVE_PROMPT,
            }.get(settings.quality_mode, FLUX_BALANCED_PROMPT)
            prompt = compose_prompt(
                settings,
                f"{mode_prompt} {settings.prompt.strip()}",
            )
            custom_negative = settings.negative_prompt.strip()
            avoid = (
                f"{DEFAULT_NEGATIVE_PROMPT}, {custom_negative}"
                if custom_negative
                else DEFAULT_NEGATIVE_PROMPT
            )
            prompt = f"{prompt} Avoid these defects: {avoid}."
            # Keep image conditioning in every mode. The uploaded workpiece
            # photographs provide the desired near-overhead inspection-camera
            # composition. Text-only creative generation tends to fall back to
            # a low, three-quarter product-photo angle even when the prompt asks
            # for 85--90 degrees. Creative mode still requests a redesigned
            # housing, but the reference anchors camera elevation and framing.
            text_only_recompose = False
            prepared = prepare_reference(reference, settings.framing, size=1024)
            steps = 4
            generator = self._torch.Generator(device="cuda").manual_seed(settings.seed)
            result = self._pipe(
                image=prepared,
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=1.0,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
            return result.convert("RGB"), {
                "effective_steps": steps,
                "effective_guidance_scale": 1.0,
                "reference_conditioning": (
                    "text_only_recompose"
                    if text_only_recompose
                    else "native_image_edit"
                ),
            }


def _unload(generator: Any) -> None:
    pipe = getattr(generator, "_pipe", None)
    if pipe is None:
        return
    generator._pipe = None
    generator._torch = None
    if hasattr(generator, "_cv2"):
        generator._cv2 = None
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_diffusers_generator = DiffusersGenerator()
_sdxl_controlnet_generator = SDXLControlNetGenerator()
_flux2_klein_generator = Flux2KleinGenerator()
_mock_generator = MockGenerator()


def generate_image(
    provider: str,
    reference_path: Path,
    settings: GenerationSettings,
) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(reference_path) as source:
        reference = source.copy()
    _, scene_variant = resolve_scene_prompt(settings.scene_preset, settings.seed)
    if provider == "mock":
        return _mock_generator.generate(reference, settings), {
            "provider": "mock",
            "training_eligible": False,
            "quality_mode": settings.quality_mode,
            "scene_preset": settings.scene_preset,
            "scene_variant": scene_variant,
        }
    if provider == "sdxl_controlnet":
        _unload(_flux2_klein_generator)
        image, effective = _sdxl_controlnet_generator.generate(reference, settings)
        return image, {
            "provider": "sdxl_controlnet",
            "training_eligible": True,
            "quality_mode": settings.quality_mode,
            "scene_preset": settings.scene_preset,
            "scene_variant": scene_variant,
            **effective,
            **_sdxl_controlnet_generator.runtime,
        }
    if provider == "flux2_klein":
        _unload(_diffusers_generator)
        _unload(_sdxl_controlnet_generator)
        image, effective = _flux2_klein_generator.generate(reference, settings)
        return image, {
            "provider": "flux2_klein",
            "training_eligible": True,
            "quality_mode": settings.quality_mode,
            "scene_preset": settings.scene_preset,
            "scene_variant": scene_variant,
            **effective,
            **_flux2_klein_generator.runtime,
        }
    _unload(_flux2_klein_generator)
    image = _diffusers_generator.generate(reference, settings)
    effective_strength = settings.strength
    effective_adapter_scale = settings.ip_adapter_scale
    effective_steps = settings.steps
    if settings.quality_mode == "strict":
        effective_strength = min(effective_strength, 0.38)
        effective_adapter_scale = max(effective_adapter_scale, 0.82)
        effective_steps = max(effective_steps, 32)
    elif settings.quality_mode == "shape_variation":
        effective_strength = max(0.50, min(effective_strength, 0.68))
        effective_adapter_scale = max(0.58, min(effective_adapter_scale, 0.76))
        effective_steps = max(effective_steps, 36)
    elif settings.quality_mode == "balanced":
        effective_strength = min(effective_strength, 0.50)
        effective_adapter_scale = max(effective_adapter_scale, 0.68)
    return image, {
        "provider": "diffusers",
        "training_eligible": True,
        "quality_mode": settings.quality_mode,
        "scene_preset": settings.scene_preset,
        "scene_variant": scene_variant,
        "effective_strength": effective_strength,
        "effective_ip_adapter_scale": effective_adapter_scale,
        "effective_steps": effective_steps,
        **_diffusers_generator.runtime,
    }
