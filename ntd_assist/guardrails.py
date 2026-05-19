from .schema import ClinicalAnalysis


def apply_morphology_guardrails(res: ClinicalAnalysis, sample_type: str = "") -> ClinicalAnalysis:
    """
    Apply morphology-based guardrails including sample-type validation.
    Catches biologically impossible combinations.
    """
    # FILARIA vs TRYPANOSOME (blood smear branch)
    if sample_type.lower().startswith("blood smear"):
        findings = (res.findings + " " + res.morphology_proof).lower()

        if "microfilaria" in findings or "sheathed" in findings:
            res = res.model_copy(update={
                "detected_disease": "Filariasis",
                "species": "Wuchereria bancrofti",
            })
        elif any(x in findings for x in ["undulating membrane", "free flagellum", "kinetoplast"]):
            res = res.model_copy(update={"detected_disease": "Trypanosomiasis"})
        else:
            res = res.model_copy(update={
                "confidence": "Medium",
                "recommendation": res.recommendation + (
                    " Morphology is ambiguous between microfilaria and trypanosome; "
                    "evaluate sheath, nuclear pattern, and tail morphology."
                ),
            })

    if "thick" in sample_type.lower():
        if res.detected_disease == "Trypanosomiasis":
            res = res.model_copy(update={
                "confidence": "Medium",
                "recommendation": res.recommendation + (
                    " Thick blood smears are more commonly used for microfilariae detection; "
                    "consider Filariasis if sheath or nuclear column is identified."
                ),
            })

    t = (res.morphology_proof + " " + res.findings + " " +
         res.observed_organisms + " " + res.observed_background).lower()

    sample_lower = sample_type.lower()
    is_tissue_sample = any(x in sample_lower for x in ["tissue", "biopsy", "skin snip", "bone marrow", "lymph"])
    is_blood_sample = any(x in sample_lower for x in ["blood", "smear"])
    is_excreta_sample = any(x in sample_lower for x in ["stool", "urine"])
    is_csf_sample = "csf" in sample_lower

    says_in_rbc = any(p in t for p in [
        "within the red blood", "within red blood", "inside red blood",
        "within rbc", "inside rbc", "intracellular", "within the rbc",
        "contained within", "inside the erythrocyte", "intraerythrocytic",
        "inside rbc", "in rbc"
    ])
    says_in_macrophage = any(p in t for p in [
        "within macrophage", "inside macrophage", "in macrophage",
        "within histiocyte", "macrophage cytoplasm", "parasitophorous",
        "intracytoplasmic", "inside large cell", "within large cell"
    ])
    says_extracellular = any(p in t for p in [
        "extracellular", "in plasma", "free in", "between cells",
        "free-swimming", "in the plasma", "between rbc"
    ])

    says_ring = any(p in t for p in ["ring form", "ring-form", "signet ring", "delicate ring", "rings with"])
    says_crescent = any(p in t for p in ["crescent", "banana", "banana-shaped", "crescentic"])
    says_schizont = "schizont" in t or "merozoite" in t
    says_trophozoite = "trophozoite" in t
    says_gametocyte = "gametocyte" in t
    malaria_indicators = says_ring or says_crescent or says_schizont or says_gametocyte

    says_amastigote = any(p in t for p in ["amastigote", "ld bod", "leishman-donovan", "oval bod"])
    says_amastigote_size = any(p in t for p in ["2-4 μm", "2–4 μm", "tiny", "much smaller than rbc"])
    says_small_oval_in_macro = says_in_macrophage and says_amastigote_size
    leishmania_indicators = says_amastigote or says_small_oval_in_macro

    says_flagellum = "flagell" in t
    says_undulating = "undulating membrane" in t
    says_trypomastigote = "trypomastigote" in t
    tryp_indicators = says_flagellum or says_undulating or says_trypomastigote

    says_microfilaria = "microfilar" in t
    says_sheathed = "sheathed" in t
    says_unsheathed = any(p in t for p in ["unsheathed", "no sheath"])
    says_larval = any(p in t for p in ["larva", "worm-like", "long thin"])
    microfilaria_indicators = says_microfilaria or (says_larval and says_extracellular)

    says_egg = "egg" in t
    says_spine = any(p in t for p in ["spine", "terminal spine", "lateral spine"])
    says_miracidium = "miracidium" in t
    schisto_indicators = says_egg and says_spine

    says_multiple_rings = any(p in t for p in ["multiple rings", "multiple per rbc", "appliqué"])
    says_schuffner = "schüffner" in t or "schuffner" in t
    says_band_form = "band form" in t
    says_c_shaped = "c-shaped" in t or "c shaped" in t
    says_tail_nuclei = any(p in t for p in ["tail nuclei", "nuclei in tail"])
    says_blunt_tail = "blunt tail" in t
    says_pointed_tail = "pointed tail" in t

    # RULE 1: SAMPLE-TYPE IMPOSSIBILITIES
    if is_tissue_sample and res.detected_disease == "Malaria" and says_in_rbc and not is_blood_sample:
        if says_in_macrophage or says_amastigote or leishmania_indicators:
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Tissue sample with intracellular organisms in macrophages suggests Leishmaniasis. Confirm with PCR or culture.",
                "morphology_proof": f"Organisms observed inside macrophages in tissue section. Original description: {res.morphology_proof}"
            })
        else:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": f"Tissue biopsy reported as Malaria with RBC findings is inconsistent. Malaria is diagnosed on blood smears. Re-evaluate the sample type and findings.",
                "morphology_proof": f"Sample-type mismatch: {sample_type} is not appropriate for malaria ring form diagnosis."
            })

    if "bone marrow" in sample_lower and res.detected_disease == "Malaria":
        if says_in_macrophage or "macrophage" in t or "histiocyte" in t:
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania donovani",
                "recommendation": "Bone marrow with intracellular organisms in macrophages is classic for Visceral Leishmaniasis. Confirm with rK39 serology or PCR.",
                "morphology_proof": "Amastigotes identified within bone marrow macrophages."
            })
        if any(p in t for p in ["small", "oval", "round bodies", "clusters"]):
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Bone marrow aspirate with small intracellular organisms suggests Visceral Leishmaniasis. Peripheral blood is preferred for malaria diagnosis.",
            })

    if res.detected_disease == "Schistosomiasis" and not is_excreta_sample and not schisto_indicators:
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Schistosomiasis requires eggs in urine/stool. Re-evaluate for microfilariae if worm-like."
        })

    if res.detected_disease == "Onchocerciasis" and not "skin" in sample_lower:
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Onchocerciasis microfilariae are in skin snips. Check sample type."
        })

    # RULE 2: LEISHMANIASIS POSITIVE IDENTIFICATION
    if res.detected_disease != "Leishmaniasis" and leishmania_indicators:
        if is_tissue_sample or says_in_macrophage:
            species = "Leishmania donovani" if any(x in sample_lower for x in ["bone", "spleen"]) else "Leishmania spp."
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Amastigotes in macrophages/tissue indicate Leishmaniasis. Speciate with PCR. Assess for visceral involvement if bone marrow/spleen positive."
            })

    # RULE 3: MALARIA VALIDATION
    if res.detected_disease == "Malaria":
        if is_tissue_sample and not is_blood_sample:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": f"Malaria diagnosis on {sample_type} is unusual. Use peripheral blood smear for malaria diagnosis."
            })

        if not (says_in_rbc or malaria_indicators):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Malaria diagnosis requires identification of intraerythrocytic parasites (rings, trophozoites, schizonts, or gametocytes)."
            })

        if res.species == "Unknown":
            if says_multiple_rings or says_crescent:
                res = res.model_copy(update={"species": "P. falciparum"})
            elif says_schuffner or says_band_form:
                res = res.model_copy(update={"species": "P. vivax" if says_schuffner else "P. malariae"})

    # RULE 4: TRYPANOSOMIASIS VALIDATION
    if res.detected_disease == "Trypanosomiasis":
        if says_in_rbc and not says_extracellular:
            if malaria_indicators or says_in_rbc:
                return res.model_copy(update={
                    "detected_disease": "Unclear",
                    "confidence": "Low",
                    "recommendation": "Trypanosomes are extracellular. Organisms inside RBCs suggest Malaria instead."
                })

        if says_in_macrophage and not says_extracellular:
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Organisms inside macrophages indicate Leishmaniasis, not Trypanosomiasis."
            })

        if says_c_shaped:
            res = res.model_copy(update={"species": "Trypanosoma cruzi"})
        elif is_csf_sample:
            res = res.model_copy(update={"species": "Trypanosoma brucei"})

    # RULE 5: FILARIASIS VALIDATION
    if res.detected_disease == "Filariasis":
        if says_in_rbc or (says_flagellum and not says_microfilaria):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Microfilariae are extracellular larvae without flagella. Re-examine for correct identification."
            })

        if says_sheathed and says_tail_nuclei:
            res = res.model_copy(update={"species": "Brugia malayi"})
        elif says_sheathed:
            res = res.model_copy(update={"species": "Wuchereria bancrofti"})

    # RULE 5.1: SCHISTOSOMIASIS VALIDATION
    if res.detected_disease == "Schistosomiasis":
        if not schisto_indicators:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Schistosomiasis requires eggs with spines. Check for other helminths."
            })

        if "terminal spine" in t:
            res = res.model_copy(update={"species": "Schistosoma haematobium"})
        elif "lateral spine" in t:
            res = res.model_copy(update={"species": "Schistosoma mansoni"})

    # RULE 5.2: ONCHOCERCIASIS / LOIASIS VALIDATION
    if res.detected_disease in ["Onchocerciasis", "Loiasis"]:
        if not microfilaria_indicators:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Requires microfilariae. Re-examine sheath and tail."
            })

        if res.detected_disease == "Onchocerciasis" and not says_unsheathed and not says_blunt_tail:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Onchocerciasis microfilariae are unsheathed with blunt tail."
            })

        if res.detected_disease == "Loiasis" and not says_sheathed and not says_pointed_tail:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Loiasis microfilariae are sheathed with pointed tail and continuous nuclei."
            })

    # RULE 6: CONSERVATIVE UNCLEAR PROMOTION
    if res.detected_disease == "Unclear":
        if microfilaria_indicators and says_unsheathed and says_blunt_tail and "skin" in sample_lower:
            return res.model_copy(update={
                "detected_disease": "Onchocerciasis",
                "confidence": "Medium",
                "species": "Onchocerca volvulus",
                "recommendation": "Unsheathed microfilariae in skin suggest Onchocerciasis."
            })

        if microfilaria_indicators and says_sheathed and says_pointed_tail and is_blood_sample:
            return res.model_copy(update={
                "detected_disease": "Loiasis",
                "confidence": "Medium",
                "species": "Loa loa",
                "recommendation": "Sheathed microfilariae with pointed tail suggest Loiasis."
            })

        if tryp_indicators and says_extracellular and not says_in_rbc and not says_in_macrophage:
            if (is_blood_sample or is_csf_sample) and not leishmania_indicators:
                species = "Trypanosoma cruzi" if says_c_shaped else "Trypanosoma brucei"
                return res.model_copy(update={
                    "detected_disease": "Trypanosomiasis",
                    "confidence": "Medium",
                    "species": species,
                    "recommendation": "Extracellular flagellated organisms suggest Trypanosomiasis. Confirm species with concentration techniques."
                })

        if is_tissue_sample and (says_in_macrophage or says_amastigote):
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Intracellular organisms in tissue macrophages suggest Leishmaniasis."
            })

        if schisto_indicators and is_excreta_sample:
            species = "Schistosoma haematobium" if "terminal spine" in t else "Schistosoma mansoni" if "lateral spine" in t else "Schistosoma spp."
            return res.model_copy(update={
                "detected_disease": "Schistosomiasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Eggs with spines in excreta suggest Schistosomiasis. Confirm with Kato-Katz."
            })

        if microfilaria_indicators and says_sheathed and is_blood_sample:
            species = "Brugia malayi" if says_tail_nuclei else "Wuchereria bancrofti"
            return res.model_copy(update={
                "detected_disease": "Filariasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Sheathed microfilariae in blood suggest lymphatic Filariasis."
            })

    # RULE 7: NEGATIVE VALIDATION
    if res.detected_disease == "Negative for Parasites":
        if res.confidence != "High":
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "severity": "N/A",
                "recommendation": "Cannot confidently exclude parasites. Examine additional fields."
            })
        if any(p in t for p in ["organism", "parasite", "seen", "observed", "identified", "present"]) and "no " not in t and "none" not in t:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Findings mention structures but diagnosis is negative. Manual review required."
            })
        if not any(x in t for x in ["200", "fields examined", "hpf", "systematically scanned", "high power fields"]):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Negative diagnosis requires examining at least 200 HPFs. Use 'Unclear' if examination extent not documented."
            })
        if not any(x in t for x in ["looked for", "searched for", "specifically examined", "malaria", "leishmania", "trypanosoma", "schistosoma", "filaria", "oncho", "loa"]):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Negative diagnosis requires stating all supported parasites were specifically looked for. Document search targets."
            })
        if not any(x in t for x in ["adequate staining", "good quality", "proper focus", "well-stained", "clear visualization"]):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Negative diagnosis requires confirming adequate staining and focus. Document slide quality."
            })

    # RULE 8: SIZE-AWARE GUARDRAIL
    if says_in_macrophage and any(p in t for p in ["rbc-sized", "same size as rbc", "7 μm", "7um", "similar to rbc"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "RBC-sized structures inside macrophages cannot be amastigotes (2-4 μm). Re-evaluate identification.",
            "morphology_proof": "Size mismatch: described structures are RBC-sized, but amastigotes are much smaller (2-4 μm)."
        })

    if schisto_indicators and any(p in t for p in ["small egg", "<50 μm", "tiny egg"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Schistosome eggs are 100-150μm. Small eggs may indicate other helminths."
        })

    if microfilaria_indicators and any(p in t for p in ["short larva", "<100 μm", "tiny worm"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Microfilariae are >200μm. Re-evaluate for protozoa if smaller."
        })

    return res
