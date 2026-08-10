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
    "deformed threads, malformed fastener, asymmetrical screw head, smooth round "
    "metal cap presented as a screw, featureless circular fastener head, blank-top "
    "external hex bolt, smooth machined bolt head, shallow engraved ring instead of "
    "a drive recess, circular outline without a cavity, center dot instead of a drive "
    "recess, smooth raised cylindrical boss, smooth vertical post, blank cylindrical "
    "cap, raised dowel, all holes and no installed screws, zero visible screws, "
    "empty threaded holes used instead of screws, more than four installed screws, "
    "screw-covered workpiece, upside-down screw, inverted screw, vertical standing screw, "
    "loose screw beside a hole, screws on the conveyor, screws on the fixture, loose "
    "bolts in the background, fasteners outside the workpiece, dense fastener pattern, dense rows of holes, dense hole "
    "grid, more than eight small holes on the top face, clustered screws, adjacent screw "
    "heads, headless threaded stud, protruding "
    "threaded rod, bare male thread above the workpiece, set screw without a visible "
    "drive head, rivet, cylindrical plug mistaken for a screw, blurry, "
    "low resolution, watermark, text"
)

FASTENER_REALISM_PROMPT = (
    "Use varied but mechanically authentic industrial screws and bolts. Every visible "
    "fastener intended as a screw must have a clearly recessed drive cavity in the "
    "center of its top face; this requirement also applies to hexagonal outer heads. "
    "Each generated workpiece must visibly contain both empty holes and installed "
    "screws: approximately 4 to 8 mechanically plausible drilled or threaded holes, "
    "with broad empty cast surfaces between them. The large central bearing opening "
    "is a single flange feature and is not counted as a small hole. Avoid dense rows "
    "or grids of holes and never exceed 8 small holes on the top face. "
    "and a sparse, non-dense arrangement of approximately 2 to 4 clearly visible "
    "screws, scaled naturally to the housing size. The exact count may vary between "
    "images; prioritize broad empty spacing and never cluster screw heads. Never "
    "satisfy the screw requirement with empty threaded holes. Use one or two drive "
    "styles within one image and rotate the drive-style selection across the batch. "
    "Use a varied mix of deep internal-hex sockets, deep six-lobe Torx sockets, "
    "recessed Phillips crosses and recessed straight slots. Vary the outer profile "
    "among cylindrical socket-cap, button-head, countersunk, flange and hexagonal "
    "combination heads, but never use a conventional blank-top external hex bolt. "
    "The recess must have visible depth, crisp edges and a contact shadow: a shallow "
    "engraved circle, concentric ring, tiny center dot or painted symbol is not a drive "
    "recess. Do not place any smooth raised cylindrical boss, plug, cap, post or dowel "
    "anywhere on the workpiece because it can be confused with a screw. Every raised "
    "cylindrical feature must instead be either an open bore or bearing seat with a "
    "clearly visible deep opening, or a fastener with one of the required recessed "
    "drives. Every screw must be fully seated in a matching hole or counterbore on the "
    "workpiece, with the head facing upward and the shaft hidden inside the hole. Never "
    "place a screw beside a hole, upside down, vertically standing, floating, or partially "
    "embedded. Never create a headless threaded stud, protruding threaded rod, exposed "
    "set screw or bare male thread on the workpiece: every installed fastener must "
    "terminate in a clearly visible recessed-drive head. Keep each fastener straight, "
    "correctly scaled and seated flush against a machined surface or washer, with a "
    "small realistic contact shadow. Do not fuse screws "
    "into the casting and do not create floating, melted or decorative fasteners. "
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
    "Use an unseen high-mounted factory inspection camera outside the frame. "
    "Keep the optical axis almost perpendicular to the work surface at an 88 to "
    "90 degree elevation. This compensates for the model's tendency to generate "
    "angles that are too low. Keep a realistic camera-to-workpiece working distance "
    "of 70 to 90 cm, preferably about 80 cm, with a normal lens rather than an "
    "ultra-wide lens. The top face must appear nearly rectangular with almost "
    "parallel edges and must dominate the image. Side walls must occupy less than "
    "10 percent of the visible workpiece height; do not use a three-quarter view. "
    "Nothing may hang above the workpiece and no camera, spindle, probe or "
    "inspection head may appear inside the image. "
    "Use the input reference photograph strictly as a camera-pose and framing "
    "template: match its overhead viewpoint and its small top-to-side visibility "
    "ratio, while changing the workpiece geometry, material details and scene. "
    "Completely redesign the reference as a different but functional industrial "
    "mechanical housing. Use a new silhouette, proportions, ribs, cooling fins, "
    "mounting ears and cavity layout. Keep continuous metal, circular drilled "
    "holes and rigid correctly seated fasteners."
)

FACTORY_SCENE_PROMPTS = {
    "assembly_line": (
        "Show only the horizontal top surface of a real factory assembly-line "
        "metal bench filling the frame, with cropped fixtures and safety markings "
        "around the workpiece; no distant production line or factory interior."
    ),
    "machine_enclosure": (
        "Show only a horizontal CNC pallet or machine-table surface filling the "
        "frame, with T-slots, clamps, cool task light and mild oil residue; no "
        "enclosure walls, spindle, probe or distant background."
    ),
    "maintenance_bench": (
        "Show only a horizontal used metal maintenance-bench surface filling the "
        "frame, with oil stains and a few cropped hand tools confined to the outer "
        "edges; no room interior, shelves, horizon or distant background."
    ),
    "conveyor_fixture": (
        "Show only the horizontal surface of a conveyor inspection fixture filling "
        "the frame, with cropped clamps, rollers and machined rails surrounding the "
        "workpiece; no distant conveyor line or factory interior."
    ),
    "warehouse_inspection": (
        "Show only the horizontal metal surface of a quality-inspection station "
        "filling the frame, with cropped locating blocks and mixed task lighting; "
        "no warehouse interior, shelves, walls, lamps or horizon."
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


def inset_for_outpaint(
    image: Image.Image,
    size: int = 1024,
    occupancy: float = 0.62,
) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    background = ImageOps.fit(
        source,
        (size, size),
        method=Image.Resampling.BILINEAR,
    )
    background = background.filter(ImageFilter.GaussianBlur(radius=38))
    background = ImageEnhance.Brightness(background).enhance(0.62)
    inset_size = max(1, round(size * occupancy))
    inset = ImageOps.fit(
        source,
        (inset_size, inset_size),
        method=Image.Resampling.LANCZOS,
    )
    offset = (size - inset_size) // 2
    background.paste(inset, (offset, offset))
    return background


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
            if settings.quality_mode == "creative":
                scene_prompt, _ = resolve_scene_prompt(
                    settings.scene_preset,
                    settings.seed,
                )
                prompt = (
                    f"{mode_prompt} {FASTENER_REALISM_PROMPT} {scene_prompt} "
                    f"{settings.prompt.strip()}"
                ).strip()
            else:
                prompt = compose_prompt(
                    settings,
                    f"{mode_prompt} {FASTENER_REALISM_PROMPT} "
                    f"{settings.prompt.strip()}",
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

            generation_passes = 1
            effective_framing = settings.framing
            if settings.quality_mode == "creative":
                outpaint_reference = inset_for_outpaint(
                    result,
                    size=1024,
                    occupancy=0.42,
                )
                outpaint_prompt = (
                    "Preserve the generated workpiece exactly, including its "
                    "silhouette, central flange, ribs, holes and fasteners. Do not "
                    "redesign, add or remove any mechanical feature. Keep the "
                    "workpiece centered at the smaller scale shown in the input, "
                    "occupying only 20 to 30 percent of the complete frame. "
                    "Leave broad, continuous work-surface margin on every side. Naturally "
                    "outpaint the surrounding horizontal factory work surface and "
                    "local fixture on every side. Maintain an apparent 80 to 90 "
                    "degree overhead view and show side walls only as a thin rim. "
                    "No horizon, distant room, camera, spindle, probe or lamp. "
                    f"Use this local surface setting: {scene_prompt} "
                    "Photorealistic industrial inspection photography with a "
                    "continuous, naturally extended work surface. Avoid visible "
                    f"seams, frames or blurred borders. Avoid: {avoid}."
                )
                outpaint_generator = self._torch.Generator(
                    device="cuda"
                ).manual_seed(settings.seed + 1_000_003)
                result = self._pipe(
                    image=outpaint_reference,
                    prompt=outpaint_prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=1.0,
                    num_inference_steps=steps,
                    generator=outpaint_generator,
                ).images[0]
                generation_passes = 2
                effective_framing = "two_stage_outpaint"
            return result.convert("RGB"), {
                "effective_steps": steps,
                "effective_guidance_scale": 1.0,
                "reference_conditioning": (
                    "text_only_recompose"
                    if text_only_recompose
                    else "native_image_edit"
                ),
                "effective_reference_framing": effective_framing,
                "generation_passes": generation_passes,
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
