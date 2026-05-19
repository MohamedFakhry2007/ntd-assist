import hashlib
import traceback
from collections import OrderedDict
from PIL import Image, ImageStat, ImageEnhance, ImageOps, ImageFilter

from . import config


_ENHANCE_CACHE: "OrderedDict[tuple, Image.Image]" = OrderedDict()


def _image_hash(image: Image.Image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


def enhance_image(image, sample_type, log=None):
    """
    Expert Enhancement: Uses Green-Channel separation to target chromatin.
    In Giemsa/Wright stains, parasites (purple/red) absorb Green light,
    making the Green channel the most information-dense for structure.
    Cached by (image hash, sample_type) so preview + inference don't double-process.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    cache_key = (_image_hash(image), sample_type.lower())
    cached = _ENHANCE_CACHE.get(cache_key)
    if cached is not None:
        _ENHANCE_CACHE.move_to_end(cache_key)
        return cached

    enhanced = ImageOps.autocontrast(image, cutoff=config.AUTOCONTRAST_CUTOFF)
    sample_lower = sample_type.lower()

    if "blood" in sample_lower:
        # Chromatin boost: invert green channel as sharpening mask for purple structures
        g = image.split()[1]
        structure_mask = ImageOps.invert(g)

        enhanced = ImageEnhance.Color(enhanced).enhance(config.BLOOD_COLOR_BOOST)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(config.BLOOD_CONTRAST_BOOST)

        sharpened = enhanced.filter(ImageFilter.UnsharpMask(**config.BLOOD_UNSHARP))
        enhanced = Image.composite(sharpened, enhanced, structure_mask)

    elif "tissue" in sample_lower or "biopsy" in sample_lower:
        enhanced = ImageOps.equalize(enhanced)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(config.TISSUE_CONTRAST_BOOST)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(config.TISSUE_SHARPNESS_BOOST)

    elif "skin" in sample_lower:
        base = ImageEnhance.Contrast(enhanced).enhance(config.SKIN_CONTRAST_BOOST)
        sharpened = base.filter(ImageFilter.UnsharpMask(**config.SKIN_UNSHARP))
        edge_mask = base.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_mask = ImageOps.autocontrast(edge_mask, cutoff=config.SKIN_EDGE_AUTOCONTRAST_CUTOFF)
        enhanced = Image.composite(sharpened, base, edge_mask)

    else:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(config.DEFAULT_SHARPNESS_BOOST)

    _ENHANCE_CACHE[cache_key] = enhanced
    if len(_ENHANCE_CACHE) > config.ENHANCE_CACHE_MAX:
        _ENHANCE_CACHE.popitem(last=False)
    return enhanced


def check_image_quality(image, log=None):
    """
    Expert Quality Check: Looks for blur (edge variance) and stain quality (color balance).
    """
    try:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        warnings = []

        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges.convert("L"))
        if edge_stat.var[0] < config.BLUR_EDGE_VAR_MIN:
            warnings.append("Blurry/Out of Focus")

        if stat.mean[0] < config.UNDEREXPOSED_MEAN_MAX:
            warnings.append("Too Dark (Underexposed)")
        if stat.mean[0] > config.OVEREXPOSED_MEAN_MIN:
            warnings.append("Overexposed (Washed out)")

        r, g, b = image.split()
        mean_r = ImageStat.Stat(r).mean[0]
        mean_b = ImageStat.Stat(b).mean[0]

        if abs(mean_r - mean_b) < config.GRAYSCALE_RB_DIFF_MAX:
            warnings.append("Low Color Information (Possible Grayscale?)")

        if log:
            log("IMAGE_QUALITY", f"Mean: {stat.mean[0]:.0f}, EdgeVar: {edge_stat.var[0]:.0f}")
        return warnings
    except Exception:
        if log:
            log("QUALITY_ERROR", traceback.format_exc())
        return ["Quality check failed — image may be corrupt"]
