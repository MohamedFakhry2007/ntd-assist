from vlm_guard.image.quality import check_image_quality as _check_image_quality
from vlm_guard.plugins.ntd_microscopy.enhance import enhance_ntd_image


def check_image_quality(image, log=None):
    warnings = _check_image_quality(image)
    if log:
        log("IMAGE_QUALITY", f"Warnings: {warnings}")
    return warnings

_ENHANCE_CACHE = {}


def enhance_image(image, sample_type, log=None):
    key = (id(image), sample_type)
    if key in _ENHANCE_CACHE:
        return _ENHANCE_CACHE[key]
    result = enhance_ntd_image(image, sample_type)
    _ENHANCE_CACHE[key] = result
    if len(_ENHANCE_CACHE) > 4:
        _ENHANCE_CACHE.pop(next(iter(_ENHANCE_CACHE)))
    if log:
        log("IMAGE_ENHANCED", f"Applied NTD enhancement for {sample_type}")
    return result
