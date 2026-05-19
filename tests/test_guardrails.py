"""Unit tests for apply_morphology_guardrails.

The function is a pure transform over ClinicalAnalysis. No model, no Streamlit.
"""
from ntd_assist.guardrails import apply_morphology_guardrails
from ntd_assist.schema import ClinicalAnalysis


def make_result(**overrides) -> ClinicalAnalysis:
    defaults = dict(
        detected_disease="Unclear",
        severity="N/A",
        morphology_proof="",
        confidence="Medium",
        findings="",
        recommendation="",
        species="Unknown",
        observed_background="",
        observed_organisms="",
        organism_location="",
    )
    return ClinicalAnalysis(**{**defaults, **overrides})


# ------------------------------------------------------------------
# Blood-smear branch (top of function)
# ------------------------------------------------------------------

def test_blood_smear_microfilaria_forces_filariasis():
    res = make_result(detected_disease="Malaria", findings="microfilaria seen")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Filariasis"
    assert out.species == "Wuchereria bancrofti"


def test_blood_smear_undulating_membrane_forces_trypanosomiasis():
    res = make_result(detected_disease="Malaria",
                      findings="undulating membrane extracellular flagellum")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Trypanosomiasis"


def test_blood_smear_ambiguous_appends_recommendation():
    # The else branch in the blood-smear block appends an ambiguity note
    # whenever findings lack microfilaria/sheathed/flagellum cues.
    res = make_result(detected_disease="Malaria",
                      findings="ring form inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert "ambiguous" in out.recommendation.lower()


# ------------------------------------------------------------------
# Thick smear branch
# ------------------------------------------------------------------

def test_thick_smear_trypanosomiasis_appends_filariasis_hint():
    res = make_result(detected_disease="Trypanosomiasis",
                      confidence="High",
                      findings="extracellular organism")
    out = apply_morphology_guardrails(res, "Blood Smear (Thick)")
    assert out.confidence == "Medium"
    assert "Thick blood smears" in out.recommendation


# ------------------------------------------------------------------
# RULE 1: Sample-type impossibilities
# ------------------------------------------------------------------

def test_tissue_malaria_with_macrophage_becomes_leishmaniasis():
    res = make_result(detected_disease="Malaria",
                      findings="parasites inside RBC inside macrophage")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Leishmaniasis"
    assert out.species == "Leishmania spp."


def test_tissue_malaria_with_rbc_only_becomes_unclear():
    res = make_result(detected_disease="Malaria",
                      findings="ring form inside RBC")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Unclear"
    assert "Sample-type mismatch" in out.morphology_proof


def test_bone_marrow_malaria_with_macrophage_becomes_visceral_leishmania():
    res = make_result(detected_disease="Malaria",
                      findings="organisms in macrophage")
    out = apply_morphology_guardrails(res, "Bone Marrow Aspirate")
    assert out.detected_disease == "Leishmaniasis"
    assert out.species == "Leishmania donovani"


def test_bone_marrow_malaria_with_small_clusters_becomes_leishmania():
    res = make_result(detected_disease="Malaria",
                      findings="small oval clusters of bodies")
    out = apply_morphology_guardrails(res, "Bone Marrow Aspirate")
    assert out.detected_disease == "Leishmaniasis"


def test_schistosomiasis_without_excreta_becomes_unclear():
    res = make_result(detected_disease="Schistosomiasis", findings="no eggs seen")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Unclear"


def test_onchocerciasis_off_skin_becomes_unclear():
    res = make_result(detected_disease="Onchocerciasis", findings="microfilaria")
    out = apply_morphology_guardrails(res, "Stool Sample")
    assert out.detected_disease == "Unclear"


# ------------------------------------------------------------------
# RULE 2: Leishmaniasis positive identification
# ------------------------------------------------------------------

def test_amastigote_in_bone_marrow_promotes_to_l_donovani():
    res = make_result(detected_disease="Unclear",
                      findings="amastigote in macrophage")
    out = apply_morphology_guardrails(res, "Bone Marrow Aspirate")
    assert out.detected_disease == "Leishmaniasis"
    assert out.species == "Leishmania donovani"


# ------------------------------------------------------------------
# RULE 3: Malaria validation
# ------------------------------------------------------------------

def test_malaria_on_non_blood_tissue_becomes_unclear():
    # Skin Snip — is_tissue_sample but not is_blood_sample, no RBC mention so
    # rule 1 doesn't fire; rule 3 catches it.
    res = make_result(detected_disease="Malaria",
                      findings="ring form")
    out = apply_morphology_guardrails(res, "Skin Snip")
    assert out.detected_disease == "Unclear"


def test_malaria_without_rbc_or_morphology_becomes_unclear():
    res = make_result(detected_disease="Malaria", findings="")
    # Use a non-blood, non-tissue sample so rule 1 / first rule 3 branch don't fire
    out = apply_morphology_guardrails(res, "Other/Unknown")
    assert out.detected_disease == "Unclear"


def test_malaria_multiple_rings_infers_p_falciparum():
    res = make_result(detected_disease="Malaria",
                      findings="ring form, multiple rings inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    # Blood-smear branch falls through (no microfilaria/sheathed/flagellum keywords)
    assert out.detected_disease == "Malaria"
    assert out.species == "P. falciparum"


def test_malaria_schuffner_infers_p_vivax():
    res = make_result(detected_disease="Malaria",
                      findings="ring form schuffner dots inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Malaria"
    assert out.species == "P. vivax"


def test_malaria_band_form_infers_p_malariae():
    res = make_result(detected_disease="Malaria",
                      findings="ring form band form inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Malaria"
    assert out.species == "P. malariae"


# ------------------------------------------------------------------
# RULE 4: Trypanosomiasis validation
# ------------------------------------------------------------------

def test_trypanosomiasis_inside_rbc_becomes_unclear():
    res = make_result(detected_disease="Trypanosomiasis",
                      findings="organism inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Unclear"


def test_trypanosomiasis_inside_macrophage_becomes_leishmaniasis():
    res = make_result(detected_disease="Trypanosomiasis",
                      findings="organism inside macrophage")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Leishmaniasis"


def test_trypanosomiasis_c_shaped_infers_t_cruzi():
    res = make_result(detected_disease="Trypanosomiasis",
                      findings="c-shaped extracellular flagellum")
    out = apply_morphology_guardrails(res, "CSF (Cerebrospinal Fluid)")
    assert out.species == "Trypanosoma cruzi"


def test_trypanosomiasis_in_csf_defaults_to_t_brucei():
    res = make_result(detected_disease="Trypanosomiasis",
                      findings="extracellular flagellum")
    out = apply_morphology_guardrails(res, "CSF (Cerebrospinal Fluid)")
    assert out.species == "Trypanosoma brucei"


# ------------------------------------------------------------------
# RULE 5: Filariasis validation
# ------------------------------------------------------------------

def test_filariasis_inside_rbc_becomes_unclear():
    res = make_result(detected_disease="Filariasis",
                      findings="microfilaria inside RBC")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.detected_disease == "Unclear"


def test_filariasis_sheathed_with_tail_nuclei_infers_b_malayi():
    res = make_result(detected_disease="Filariasis",
                      findings="sheathed nuclei in tail")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.species == "Brugia malayi"


def test_filariasis_sheathed_without_tail_nuclei_infers_w_bancrofti():
    res = make_result(detected_disease="Filariasis",
                      findings="sheathed microfilaria")
    out = apply_morphology_guardrails(res, "Blood Smear (Thin)")
    assert out.species == "Wuchereria bancrofti"


# ------------------------------------------------------------------
# RULE 5.1: Schistosomiasis species
# ------------------------------------------------------------------

def test_schistosomiasis_terminal_spine_infers_haematobium():
    res = make_result(detected_disease="Schistosomiasis",
                      findings="egg with terminal spine")
    out = apply_morphology_guardrails(res, "Urine Sediment")
    assert out.detected_disease == "Schistosomiasis"
    assert out.species == "Schistosoma haematobium"


def test_schistosomiasis_lateral_spine_infers_mansoni():
    res = make_result(detected_disease="Schistosomiasis",
                      findings="egg with lateral spine")
    out = apply_morphology_guardrails(res, "Stool Sample")
    assert out.species == "Schistosoma mansoni"


# ------------------------------------------------------------------
# RULE 5.2: Onchocerciasis / Loiasis
# ------------------------------------------------------------------

def test_onchocerciasis_without_unsheathed_or_blunt_becomes_unclear():
    res = make_result(detected_disease="Onchocerciasis",
                      findings="microfilaria present")
    out = apply_morphology_guardrails(res, "Skin Snip")
    assert out.detected_disease == "Unclear"


def test_loiasis_without_sheathed_or_pointed_becomes_unclear():
    res = make_result(detected_disease="Loiasis",
                      findings="microfilaria present")
    out = apply_morphology_guardrails(res, "Skin Snip")
    assert out.detected_disease == "Unclear"


# ------------------------------------------------------------------
# RULE 6: Unclear promotion
# ------------------------------------------------------------------

def test_unclear_with_flagellate_in_csf_promotes_to_trypanosomiasis():
    res = make_result(detected_disease="Unclear",
                      findings="extracellular flagellum")
    out = apply_morphology_guardrails(res, "CSF (Cerebrospinal Fluid)")
    assert out.detected_disease == "Trypanosomiasis"
    assert out.species == "Trypanosoma brucei"


def test_unclear_with_tissue_macrophage_promotes_to_leishmaniasis():
    res = make_result(detected_disease="Unclear",
                      findings="organisms inside macrophage")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Leishmaniasis"


def test_unclear_unsheathed_microfilaria_in_skin_promotes_to_onchocerciasis():
    # Regression: this promotion was previously dead code (nested under a never-true branch).
    res = make_result(detected_disease="Unclear",
                      findings="microfilaria unsheathed blunt tail")
    out = apply_morphology_guardrails(res, "Skin Snip")
    assert out.detected_disease == "Onchocerciasis"
    assert out.species == "Onchocerca volvulus"


def test_unclear_sheathed_microfilaria_in_blood_promotes_to_loiasis():
    # Regression: previously dead code.
    # Must avoid the blood-smear branch (which converts "sheathed" → Filariasis),
    # so use a sample containing "blood" but not starting with "blood smear".
    res = make_result(detected_disease="Unclear",
                      findings="microfilaria sheathed pointed tail")
    out = apply_morphology_guardrails(res, "Peripheral Blood")
    assert out.detected_disease == "Loiasis"
    assert out.species == "Loa loa"


def test_unclear_csf_flagellate_with_leishmania_indicators_does_not_promote():
    # Regression for operator-precedence bug: with leishmania_indicators present
    # in a CSF sample, the Trypanosomiasis promotion must NOT fire.
    res = make_result(detected_disease="Unclear",
                      findings="extracellular flagellum amastigote present")
    out = apply_morphology_guardrails(res, "CSF (Cerebrospinal Fluid)")
    assert out.detected_disease != "Trypanosomiasis"


def test_unclear_with_eggs_in_excreta_promotes_to_schistosomiasis():
    res = make_result(detected_disease="Unclear",
                      findings="egg with lateral spine")
    out = apply_morphology_guardrails(res, "Stool Sample")
    assert out.detected_disease == "Schistosomiasis"
    assert out.species == "Schistosoma mansoni"


# ------------------------------------------------------------------
# RULE 7: Negative validation
# ------------------------------------------------------------------

def test_negative_with_medium_confidence_becomes_unclear():
    res = make_result(detected_disease="Negative for Parasites", confidence="Medium")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Unclear"


def test_negative_missing_hpf_becomes_unclear():
    res = make_result(detected_disease="Negative for Parasites",
                      confidence="High",
                      findings="looked for malaria, adequate staining confirmed")
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Unclear"
    assert "HPF" in out.recommendation


def test_negative_with_all_criteria_stays_negative():
    res = make_result(
        detected_disease="Negative for Parasites",
        confidence="High",
        findings="200 HPFs examined; looked for malaria leishmania trypanosoma; adequate staining confirmed",
    )
    out = apply_morphology_guardrails(res, "Tissue Biopsy")
    assert out.detected_disease == "Negative for Parasites"


# ------------------------------------------------------------------
# RULE 8: Size-aware guardrails
# ------------------------------------------------------------------

def test_rbc_sized_in_macrophage_becomes_unclear():
    res = make_result(detected_disease="Leishmaniasis",
                      findings="organisms inside macrophage 7 μm")
    out = apply_morphology_guardrails(res, "Bone Marrow Aspirate")
    assert out.detected_disease == "Unclear"
    assert "Size mismatch" in out.morphology_proof


def test_small_schisto_egg_becomes_unclear():
    res = make_result(detected_disease="Schistosomiasis",
                      findings="egg with spine, small egg")
    out = apply_morphology_guardrails(res, "Stool Sample")
    assert out.detected_disease == "Unclear"
