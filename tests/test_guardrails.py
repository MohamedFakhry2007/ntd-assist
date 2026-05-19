"""Smoke tests for NTD-Assist integration with vlm-guard."""

from vlm_guard.core.analysis import Analysis
from ntd_assist.guardrails import apply_morphology_guardrails
from ntd_assist.schema import build_ntd_analysis


def test_guardrails_returns_analysis():
    res = build_ntd_analysis(detected_disease="Malaria", findings="ring form inside RBC on blood smear")
    out = apply_morphology_guardrails(res, sample_type="Blood Smear (Thin)")
    assert isinstance(out, Analysis)


def test_guardrails_corrects_tissue_malaria():
    res = build_ntd_analysis(detected_disease="Malaria", findings="parasites inside RBC inside macrophage")
    out = apply_morphology_guardrails(res, sample_type="Tissue Biopsy")
    assert out.label == "Leishmaniasis"
    assert out.metadata.get("species") == "Leishmania spp."


def test_build_ntd_analysis_maps_detected_disease():
    res = build_ntd_analysis(detected_disease="Malaria")
    assert res.label == "Malaria"


def test_build_ntd_analysis_preserves_label():
    res = build_ntd_analysis(label="Malaria")
    assert res.label == "Malaria"
