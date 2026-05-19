from vlm_guard.core.analysis import Analysis
from vlm_guard.plugins.ntd_microscopy.schema import ntd_analysis_from_dict


def build_ntd_analysis(**kwargs) -> Analysis:
    if "detected_disease" in kwargs:
        kwargs["label"] = kwargs.pop("detected_disease")
    return ntd_analysis_from_dict(kwargs)
