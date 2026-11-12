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
    "fake hole, painted circle, oval hole, collapsed hole, "
    "deformed threads, malformed fastener, asymmetrical screw head, featureless "
    "circular fastener head, circular cap without a drive recess, smooth round metal "
    "smooth black screw head, smooth black cap, black round cap, black circular disk, "
    "black circular outer perimeter, black button-head screw, black ring without a hex socket, "
    "disk, circular screw head without a visible drive, shallow engraved ring instead of "
    "a drive recess, circular outline without a cavity, center dot instead of a drive "
    "recess, smooth raised cylindrical boss, smooth vertical post, blank cylindrical "
    "cap, raised dowel, upright hollow tube, hollow cylindrical standoff, open tubular boss, "
    "vertical sleeve, tube with a hole on top, raised collar without a seated screw, "
    "screw-covered workpiece, upside-down screw, inverted screw, vertical standing screw, "
    "loose screw beside a hole, loose bolt, loose nut, loose washer, detached fastener, "
    "detached cylindrical metal piece, upright metal post, threaded post, background fastener, "
    "stacked screws, overlapping screws, double-stacked fastener, "
    "two screws in one hole, fused screws, screws on the conveyor, screws on the fixture, loose "
    "bolts in the background, fasteners outside the workpiece, dense fastener pattern, dense rows of holes, dense hole "
    "grid, more than two empty holes on the top face, clustered screws, adjacent screw "
    "heads, headless threaded stud, protruding "
    "missing black socket-head screw, missing natural silver internal-hex socket-head screw, "
    "missing smooth natural-metal external-hex bolt, painted white screw, plastic white screw, "
    "all screws white, only one fastener style, complete screw, full bolt, vertical bolt, threaded cylinder, upright fastener, "
    "projecting bolt, exposed shank, screw sidewall, cylindrical screw body, visible screw shaft, exposed screw body, "
    "visible threads below head, screw neck, head on a post, screw sleeve, washer under screw, "
    "nut under screw, boss around screw, anything except the fastener head, "
    "threaded rod, bare male thread above the workpiece, set screw without a visible "
    "drive head, rivet, cylindrical plug mistaken for a screw, blurry, "
    "low resolution, watermark, text"
)

FASTENER_REALISM_PROMPT = (
    "Show 3 to 5 installed fasteners on the workpiece top face, and include all three "
    "required styles in every image, not an either-or choice: at least one matte black "
    "flush-mounted six-sided polygonal socket-head top viewed straight from above. The black "
    "fastener's OUTER PERIMETER must be visibly hexagonal or another straight-edged polygon, "
    "never circular, and its top must contain one large, deep, sharply defined internal-hex "
    "recess; a smooth black head or black round cap is forbidden. "
    "clearly visible internal-hex socket; at least one natural silver-gray stainless "
    "natural silver-gray flush-mounted recessed-hex head with the same large, deep internal-hex socket and "
    "a flat six-sided outer silhouette, never a circular perimeter, "
    "real metallic highlights, never painted white or plastic; and at least one natural "
    "silver-metal flush-mounted external-hex head with a flat six-sided top, only a thin edge bevel and a completely smooth, unbroken surface "
    "with no groove, recess, socket or cross, like the smooth silver bolt in the "
    "reference photograph, fully seated flush in a matching hole in the top face, never on a raised boss. The hex-head "
    "style is a real installed bolt, not a loose nut: no separate washer, no exposed long "
    "shaft and no nut-only appearance. Metal socket screws never use a Phillips cross. Every fastener "
    "head faces the camera, is a flat low-profile horizontal head fully seated flush or only slightly proud in one "
    "matching hole in the top face, and only the fastener head top is visible. This is a top-face inspection image, "
    "not a hardware catalog: never draw a complete screw or bolt. Show no screw shaft, threaded body, visible "
    "threads below the head, neck, shoulder below the head, sleeve, collar, tube, boss, washer or nut around or "
    "under it; the shaft is completely hidden below the workpiece surface. Keep exactly 1 to 2 additional empty "
    "drilled holes and no other visible empty holes, not counting the main bearing opening, and "
    "broad empty cast-metal surfaces. The workbench and area around the workpiece must "
    "contain zero detached screws, bolts, nuts, washers, threaded rods, studs, cylindrical "
    "metal pieces or upright posts. All fasteners must lie horizontally in the workpiece; "
    "never create a vertical screw, vertical sleeve, hollow cylindrical standoff, open tube, "
    "raised collar with a hole but no seated screw, or any tube-like boss. Never show a featureless flat disk, circular cap without a drive recess, circular ring mistaken for an empty hole, tiny socket recess, "
    "Phillips cross, isolated external hex nut, washer without a screw, upright screw, "
    "upside-down screw, exposed threaded rod, headless stud, loose or stacked fasteners, "
    "or an ambiguous hollow cylinder. "
)

STRICT_MECHANICAL_PROMPT = (
    " Preserve the reference workpiece's exact mechanical topology: keep the same "
    "number, position, diameter, circularity and depth of holes; keep every screw "
    "and bolt rigid, hexagonal, correctly seated and aligned; preserve sharp machined "
    "edges and physically continuous metal. Vary only lighting, surface finish, "
    "camera exposure and workbench background."
)

SHAPE_VARIATION_PROMPT = (
    "Use the uploaded image only as a reference for the requested fastener head style; "
    "do not copy its workpiece silhouette, proportions, central layout, ribs, holes or "
    "background. Redesign a genuinely different, mechanically plausible low-profile "
    "industrial housing. Keep holes circular and drilled; keep screws rigid, aligned "
    "and seated. Change the overall silhouette, proportions, mounting ears, ribs, "
    "cooling fins and cavity layout. The geometric center must be solid metal, a "
    "rectangular panel or an irregular rib junction, never a large circular boss, "
    "flange, bearing opening or round recess. Use continuous machined metal without "
    "fused parts."
)

# FLUX.2 tends to repeat the geometry of a conditioning image unless the requested
# topology is made explicit.  Rotate through independent housing families by seed so
# a batch of consecutive candidates cannot collapse into copies of one reference.
SHAPE_VARIATION_FAMILIES = (
    "Geometry family A: use a long shallow rectangular housing with a broad offset "
    "rectangular service panel, parallel rib channels and a short side bank of cooling "
    "fins. No circular feature near the center.",
    "Geometry family B: use a shallow trapezoidal or triangular plate-like casting with "
    "unequal corner ears, diagonal reinforcement ribs and an offset rectangular slot. "
    "The center remains a solid irregular rib junction, not a circle.",
    "Geometry family C: use an irregular pentagonal or notched polygonal housing with "
    "stepped terraces, one side-heavy fin group and a small rectangular or hexagonal "
    "shaft interface away from the center. Do not make a round central boss.",
    "Geometry family D: use a split two-level rectangular cover with a long rounded-"
    "rectangle inspection cavity, asymmetric mounting ears and non-mirrored rib grids. "
    "Keep the center flat and rectangular, with no circular opening.",
    "Geometry family E: use a compact U-shaped or forked casting with a solid crossbar, "
    "two unequal side towers and stepped machined pockets. Use only small peripheral "
    "drilled holes; no central circular flange, bearing or ring.",
)


def resolve_shape_variation_family(seed: int) -> str:
    return SHAPE_VARIATION_FAMILIES[seed % len(SHAPE_VARIATION_FAMILIES)]

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
        "The entire background must be a dirty, long-used maintenance bench. "
        "Choose either a worn yellow-beige laminated tabletop with cracks, tape "
        "residue and dark grease stains, or a scratched dark-green anti-static mat. "
        "Place only two to four cropped used hand tools near the image edges. "
        "Keep the workpiece fully visible. Do not use a silver metal table, clean "
        "machine fixture, T-slots, conveyor rails or studio surface."
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
    edge_only_background: bool = False,
) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    if edge_only_background:
        # Build the outpaint canvas only from corner workbench texture. Using a
        # blurred full image leaves a large ghost of the centered workpiece, and
        # FLUX reconstructs that ghost instead of respecting the small inset.
        patch_size = max(32, round(min(source.size) * 0.28))
        corners = (
            source.crop((0, 0, patch_size, patch_size)),
            source.crop((source.width - patch_size, 0, source.width, patch_size)),
            source.crop((0, source.height - patch_size, patch_size, source.height)),
            source.crop((
                source.width - patch_size,
                source.height - patch_size,
                source.width,
                source.height,
            )),
        )
        mosaic = Image.new("RGB", (patch_size * 2, patch_size * 2))
        mosaic.paste(corners[0], (0, 0))
        mosaic.paste(corners[1], (patch_size, 0))
        mosaic.paste(corners[2], (0, patch_size))
        mosaic.paste(corners[3], (patch_size, patch_size))
        background = mosaic.resize((size, size), Image.Resampling.BILINEAR)
    else:
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


def reframe_with_edge_texture(
    image: Image.Image,
    size: int = 1024,
    content_scale: float = 0.64,
) -> Image.Image:
    """Deterministically add continuous reflected margin around the full image."""
    source = ImageOps.exif_transpose(image).convert("RGB")
    inset_size = max(1, round(size * content_scale))
    inset = ImageOps.fit(
        source,
        (inset_size, inset_size),
        method=Image.Resampling.LANCZOS,
    )
    left = (size - inset_size) // 2
    right = size - inset_size - left
    top = left
    bottom = right
    canvas = Image.new("RGB", (size, size))
    canvas.paste(inset, (left, top))

    top_source = inset.crop((0, 0, inset_size, min(top, inset_size)))
    top_strip = ImageOps.flip(top_source).resize(
        (inset_size, top), Image.Resampling.BICUBIC
    )
    canvas.paste(top_strip, (left, 0))
    bottom_source = inset.crop((
        0,
        max(0, inset_size - bottom),
        inset_size,
        inset_size,
    ))
    bottom_strip = ImageOps.flip(bottom_source).resize(
        (inset_size, bottom), Image.Resampling.BICUBIC
    )
    canvas.paste(bottom_strip, (left, top + inset_size))

    left_source = canvas.crop((left, 0, left + min(left, inset_size), size))
    left_strip = ImageOps.mirror(left_source).resize(
        (left, size), Image.Resampling.BICUBIC
    )
    canvas.paste(left_strip, (0, 0))
    right_source = canvas.crop((
        left + max(0, inset_size - right),
        0,
        left + inset_size,
        size,
    ))
    right_strip = ImageOps.mirror(right_source).resize(
        (right, size), Image.Resampling.BICUBIC
    )
    canvas.paste(right_strip, (left + inset_size, 0))
    return canvas


def composite_central_workpiece(
    workpiece_image: Image.Image,
    background_image: Image.Image,
    size: int = 1024,
    max_dimension_fraction: float = 0.42,
) -> Image.Image:
    """Extract the central workpiece and place it on a sharp full-frame bench."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - school ML runtime provides cv2
        raise RuntimeError("OpenCV is required for maintenance-bench compositing.") from exc

    source = np.asarray(
        ImageOps.fit(
            ImageOps.exif_transpose(workpiece_image).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
    )
    mask = np.zeros((size, size), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    margin = round(size * 0.10)
    cv2.grabCut(
        cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
        mask,
        (margin, margin, size - 2 * margin, size - 2 * margin),
        bg_model,
        fg_model,
        6,
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
        iterations=2,
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        raise RuntimeError("Could not isolate the generated workpiece.")
    center = np.array([size / 2, size / 2])
    candidates: list[tuple[float, int]] = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < size * size * 0.015:
            continue
        distance = float(np.linalg.norm(centroids[label] - center)) / size
        candidates.append((area * max(0.15, 1.0 - distance * 1.8), label))
    if not candidates:
        raise RuntimeError("No central workpiece component was found.")
    selected = max(candidates)[1]
    binary = np.where(labels == selected, 255, 0).astype(np.uint8)
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
    x, y, width, height = cv2.boundingRect(binary)
    padding = max(6, round(max(width, height) * 0.025))
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(size, x + width + padding), min(size, y + height + padding)

    object_rgb = Image.fromarray(source[y1:y2, x1:x2])
    object_mask = Image.fromarray(binary[y1:y2, x1:x2]).filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )
    target = max(1, round(size * max_dimension_fraction))
    scale = min(target / object_rgb.width, target / object_rgb.height)
    new_size = (
        max(1, round(object_rgb.width * scale)),
        max(1, round(object_rgb.height * scale)),
    )
    object_rgb = object_rgb.resize(new_size, Image.Resampling.LANCZOS)
    object_mask = object_mask.resize(new_size, Image.Resampling.LANCZOS)

    canvas = ImageOps.fit(
        ImageOps.exif_transpose(background_image).convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_mask = Image.new("L", (size, size), 0)
    # A long directional shadow makes a flat overhead object appear raised and
    # pitched toward the camera. Keep only a tight, almost centred contact shadow.
    shadow_mask.paste(object_mask, (offset[0], offset[1] + 2))
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=4))
    shadow.putalpha(shadow_mask.point(lambda value: round(value * 0.20)))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow)
    canvas.paste(object_rgb, offset, object_mask)
    return canvas.convert("RGB")


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
            shape_family_prompt = (
                resolve_shape_variation_family(settings.seed)
                if settings.quality_mode == "shape_variation"
                else ""
            )
            scene_prompt, _ = resolve_scene_prompt(
                settings.scene_preset,
                settings.seed,
            )
            if (
                settings.quality_mode == "shape_variation"
                and settings.scene_preset == "maintenance_bench"
            ):
                # Including the maintenance-scene language in the workpiece pass
                # gives the housing realistic wear, color and illumination that
                # blends naturally with the separately generated clear bench.
                user_prompt = settings.prompt.strip()
                # Keep one compact fastener specification in the effective prompt.
                # The UI may already include it from the recipe; do not append it
                # repeatedly on every submission.
                if "Show 3 to 5 installed fasteners" not in user_prompt:
                    user_prompt = f"{FASTENER_REALISM_PROMPT} {user_prompt}".strip()
                # This path intentionally starts from text rather than the uploaded
                # workpiece. Include the shape-variation mode prompt here; the old
                # implementation accidentally omitted it, allowing the UI recipe
                # and reference geometry to dominate every candidate.
                prompt = (
                    f"{scene_prompt} {mode_prompt} {shape_family_prompt} {user_prompt}"
                ).strip()
            elif settings.quality_mode == "creative":
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
            # The editing pipeline strongly inherits and often multiplies the
            # reference workpiece's holes, fasteners and background. Creative mode,
            # and shape-variation on a maintenance bench, therefore start from text
            # only. A second pass supplies the requested overhead environment.
            text_only_recompose = settings.quality_mode == "creative" or (
                settings.quality_mode == "shape_variation"
                and settings.scene_preset == "maintenance_bench"
            )
            prepared = prepare_reference(reference, settings.framing, size=1024)
            steps = 4
            generator = self._torch.Generator(device="cuda").manual_seed(settings.seed)
            first_pass_args: dict[str, Any] = {
                "prompt": prompt,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 1.0,
                "num_inference_steps": steps,
                "generator": generator,
            }
            if not text_only_recompose:
                first_pass_args["image"] = prepared
            result = self._pipe(
                **first_pass_args,
            ).images[0]

            generation_passes = 1
            effective_framing = settings.framing
            if text_only_recompose:
                if settings.scene_preset == "maintenance_bench":
                    background_prompt = (
                        "Near-overhead smartphone photograph of an empty dirty, "
                        "long-used university maintenance workbench. Choose a worn "
                        "yellow-beige laminated tabletop with cracks, tape residue, "
                        "dark grease stains and scratches, or a scratched dark-green "
                        "anti-static mat. Put only two to four cropped used hand tools "
                        "near the outer image edges. Keep the central half of the bench "
                        "clear and empty for later placement of one workpiece. Full-frame "
                        "sharp focus with uniform natural detail from edge to edge. No "
                        "workpiece, gearbox, metal housing, large circular metal object, "
                        "blur, bokeh, depth-of-field blur, silver machine table, fixture, "
                        "T-slots, conveyor, studio background, text or logo."
                    )
                    background_generator = self._torch.Generator(
                        device="cuda"
                    ).manual_seed(settings.seed + 2_000_003)
                    clear_background = self._pipe(
                        prompt=background_prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=steps,
                        generator=background_generator,
                    ).images[0]
                    result = composite_central_workpiece(
                        result,
                        clear_background,
                        size=1024,
                        max_dimension_fraction=0.42,
                    )
                    # A geometrically controlled cutout solves framing, but a raw
                    # paste still has mismatched illumination, colour temperature,
                    # sensor texture and contact shadows. Use the composite as an
                    # edit reference for one final photographic harmonisation pass.
                    # The prompt deliberately forbids layout or mechanical changes;
                    # manual review remains mandatory because this is generative.
                    harmonize_prompt = (
                        "Photographically harmonize this existing maintenance-bench "
                        "inspection image into one naturally captured smartphone "
                        "photo. Preserve the exact composition, camera angle, scale, "
                        "workpiece position, outer silhouette, flange position, ribs, "
                        "holes, installed screws, tools and background layout. Do not "
                        "add, remove, duplicate, relocate or redesign any mechanical "
                        "feature or object. Only make the workpiece genuinely belong "
                        "on the bench: unify illumination direction, exposure, white "
                        "balance, local metallic reflections, fine sensor grain and "
                        "sharpness; add a subtle tight contact shadow and ambient "
                        "occlusion directly and evenly beneath its perimeter, without "
                        "a directional cast shadow. The workpiece top face must remain "
                        "parallel to the workbench in a true overhead view: opposite "
                        "edges stay parallel, circular openings remain circular, near "
                        "and far mounting holes keep equal apparent size, and no front "
                        "side wall becomes thicker than the rear wall. Retain the dull, "
                        "used aluminum machining marks, scratches, grease and light "
                        "oxidation. Keep the entire frame sharply detailed with no "
                        "bokeh, blurred sides, halo, cutout edge, floating object, "
                        "studio lighting, CGI smoothness, text or logo."
                    )
                    harmonize_generator = self._torch.Generator(
                        device="cuda"
                    ).manual_seed(settings.seed + 3_000_007)
                    result = self._pipe(
                        image=result,
                        prompt=harmonize_prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=1.0,
                        num_inference_steps=steps,
                        generator=harmonize_generator,
                    ).images[0]
                    generation_passes = 3
                    effective_framing = "harmonized_sharp_background_composite"
                    return result.convert("RGB"), {
                        "effective_steps": steps,
                        "effective_guidance_scale": 1.0,
                        "reference_conditioning": "text_only_recompose",
                        "effective_reference_framing": effective_framing,
                        "generation_passes": generation_passes,
                    }
                outpaint_reference = inset_for_outpaint(
                    result,
                    size=1024,
                    occupancy=(
                        0.18
                        if settings.scene_preset == "maintenance_bench"
                        else 0.42
                    ),
                    edge_only_background=(
                        settings.scene_preset == "maintenance_bench"
                    ),
                )
                outpaint_prompt = (
                    "Preserve the generated workpiece exactly, including its "
                    "silhouette, central flange, ribs, holes and fasteners. Do not "
                    "redesign, add or remove any mechanical feature. Keep the "
                    "workpiece centered at the smaller scale shown in the input, "
                    "occupying only 12 to 18 percent of the complete frame; this "
                    "small requested size compensates for the model enlarging it. "
                    "Leave broad work-surface margin on every side. Outpaint only "
                    "the requested dirty maintenance bench on every side. Do not "
                    "create a silver metal table or machine fixture. Maintain an 80 to 90 "
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
