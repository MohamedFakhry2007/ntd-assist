from vlm_guard import GuardrailEngine
from vlm_guard.plugins.ntd_microscopy import register_ntd_rules

_engine = GuardrailEngine()
register_ntd_rules(_engine)


def apply_morphology_guardrails(analysis, sample_type: str = ""):
    return _engine.apply(analysis, context={"sample_type": sample_type})
