import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image, ImageStat, ImageEnhance, ImageOps, ImageChops, ImageFilter
from pydantic import BaseModel
from typing import Literal
import re
import datetime
import time
from fpdf import FPDF
import traceback

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NTD-Assist",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

def log_debug(stage: str, data):
    st.session_state.debug_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "stage": stage,
        "data": str(data)[:2000]
    })

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_medgemma():
    model_id = "google/medgemma-4b-it"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        return processor, model, None
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 3. IMAGE ENHANCEMENT
# ==========================================
from PIL import ImageChops

def enhance_image(image, sample_type):
    """
    Expert Enhancement: Uses Green-Channel separation to target chromatin.
    In Giemsa/Wright stains, parasites (purple/red) absorb Green light,
    making the Green channel the most information-dense for structure.
    """
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Split channels
    r, g, b = image.split()
    
    # 1. CHROMATIN BOOST (The "Pathologist's Filter")
    # Invert Green channel to make purple objects bright, use as mask for sharpening
    structure_mask = ImageOps.invert(g)
    
    # Base enhancement
    enhanced = ImageOps.autocontrast(image, cutoff=0.5) 
    
    if "blood" in sample_type.lower():
        # Blood Smears: Highlight chromatin dots (rings)
        # Increase Saturation to separate blue cytoplasm from red chromatin
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.8) # High color boost for Giemsa
        
        # Increase Contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.4)
        
        # Adaptive Sharpening using the Green Channel info
        # Only sharpen areas that are dark in the green channel (nuclei/chromatin)
        sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        enhanced = Image.composite(sharpened, enhanced, structure_mask)

    elif "tissue" in sample_type.lower() or "biopsy" in sample_type.lower():
        # Tissue: Needs brightness balance to see inside macrophages
        enhanced = ImageOps.equalize(enhanced) # Histogram equalization helps tissue texture
        
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.3)
        
    elif "skin" in sample_type.lower():
        # Skin Snips (Onchocerca): Needs edge detection for transparency
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.5)
        enhanced = enhanced.filter(ImageFilter.FIND_EDGES) # Experimental: highlight worm outlines
        # Blend back with original to keep context
        enhanced = Image.blend(image, enhanced.convert('RGB'), alpha=0.3)

    else:
        # General backup
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)

    return enhanced

def check_image_quality(image):
    """
    Expert Quality Check: Looks for blur (Laplacian variance) 
    and stain quality (Color balance).
    """
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    warnings = []
    
    # 1. Blur Detection (Laplacian Variance)
    # A sharp microscopy image usually has variance > 500
    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges.convert("L"))
    if edge_stat.var[0] < 20: # Threshold depends on resolution, strictly low here means blur
        warnings.append("Blurry/Out of Focus")

    # 2. Exposure
    if stat.mean[0] < 40:
        warnings.append("Too Dark (Underexposed)")
    if stat.mean[0] > 240:
        warnings.append("Overexposed (Washed out)")
        
    # 3. Stain Quality (Red/Blue Ratio) - Rudimentary Giemsa check
    # Giemsa images should have significant Blue and Red channels.
    r, g, b = image.split()
    mean_r = ImageStat.Stat(r).mean[0]
    mean_b = ImageStat.Stat(b).mean[0]
    
    if abs(mean_r - mean_b) < 5: # If R and B are identical, it might be grayscale
        warnings.append("Low Color Information (Possible Grayscale?)")

    log_debug("IMAGE_QUALITY", f"Mean: {stat.mean[0]:.0f}, EdgeVar: {edge_stat.var[0]:.0f}")
    return warnings
    
# ==========================================
# 4. PDF GENERATION
# ==========================================
def create_pdf(res, sample_type, stain, magnification, context):
    """Generate PDF diagnostic report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "NTD-Assist Diagnostic Report", ln=True, align="C")
    pdf.ln(5)
    
    # Report info
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(3)
    
    # Sample details box
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Sample Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Sample Type: {sample_type}", ln=True)
    pdf.cell(0, 6, f"Stain: {stain}", ln=True)
    pdf.cell(0, 6, f"Magnification: {magnification}", ln=True)
    pdf.cell(0, 6, f"Patient Context: {context if context else 'Not provided'}", ln=True)
    pdf.ln(5)
    
    # Diagnosis box
    pdf.set_font("Arial", "B", 14)
    pdf.set_fill_color(255, 230, 230) if res.detected_disease not in ["Negative for Parasites", "Unclear"] else pdf.set_fill_color(230, 255, 230)
    pdf.cell(0, 10, f"DIAGNOSIS: {res.detected_disease}", ln=True, fill=True)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Species: {res.species}", ln=True)
    pdf.cell(0, 7, f"Severity: {res.severity}", ln=True)
    pdf.cell(0, 7, f"Confidence: {res.confidence}", ln=True)
    pdf.ln(5)
    
    # Findings
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Morphological Evidence", ln=True)
    pdf.set_font("Arial", "", 10)
    morphology = res.morphology_proof.replace('\u03bc', 'u').replace('\u03bcm', 'um')
    pdf.multi_cell(0, 6, morphology)
    pdf.ln(3)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Findings", ln=True)
    pdf.set_font("Arial", "", 10)
    findings = res.findings.replace('\u03bc', 'u').replace('\u03bcm', 'um')
    pdf.multi_cell(0, 6, findings)
    pdf.ln(3)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendation", ln=True)
    pdf.set_font("Arial", "", 10)
    recommendation = res.recommendation.replace('\u03bc', 'u').replace('\u03bcm', 'um')
    pdf.multi_cell(0, 6, recommendation)
    pdf.ln(5)
    
    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This AI-assisted analysis is for educational and screening purposes only. "
                         "All findings must be confirmed by a qualified medical professional. "
                         "Do not use as sole basis for clinical decisions.")
    
    return pdf.output(dest="S").encode("latin-1")

# ==========================================
# 5. SCHEMA
# ==========================================
class ClinicalAnalysis(BaseModel):
    detected_disease: Literal[
        "Malaria", "Leishmaniasis", "Schistosomiasis",
        "Filariasis", "Trypanosomiasis", "Onchocerciasis",
        "Loiasis", "Negative for Parasites", "Unclear"
    ]
    severity: Literal["Scanty (+)", "Moderate (++)", "Heavy (+++)", "N/A"]
    morphology_proof: str
    confidence: Literal["High", "Medium", "Low"]
    findings: str
    recommendation: str
    species: str = "Unknown"
    # New observation fields (optional for backward compatibility)
    observed_background: str = ""
    observed_organisms: str = ""
    organism_location: str = ""

def build_minimal_prompt(sample_type, magnification, stain, patient_context):
    """
    Observation-first prompt that forces the model to describe what it sees
    BEFORE attempting diagnosis. Includes sample-type specific guidance.
    """
    
    context = re.sub(r"[^\w\s.,-]", "", patient_context or "").strip()
    
    # Sample-type specific observation guidance
    sample_guidance = get_sample_specific_guidance(sample_type)
    
    prompt = f"""You are an expert parasitologist. Analyze this microscopy image by FIRST describing exactly what you observe, THEN making a diagnosis.

SAMPLE INFORMATION:
• Type: {sample_type}
• Stain: {stain}  
• Magnification: {magnification}
• Clinical Context: {context if context else "None provided"}

{sample_guidance}

═══════════════════════════════════════════════════════════
STEP 1: OBSERVATION (describe ONLY what you actually see)
═══════════════════════════════════════════════════════════

Look at the image and describe:
1. BACKGROUND:
   - Dominant cell type(s)
   - Tissue architecture or smear quality

2. ABNORMAL STRUCTURES:
   - COUNT: approximate number per high power field (HPF)
   - DISTRIBUTION: focal vs diffuse
   - LOCATION: intracellular (specify cell) vs extracellular
   - SIZE: relative to RBC or nucleus
   - INTERNAL STRUCTURES: nucleus, kinetoplast, pigment, sheath

3. If you see NO abnormal structures, state that clearly.

DO NOT use diagnostic terminology yet. Just describe what you observe.

**MORPHOLOGY CONSTRAINTS:**
When describing morphology:
- Do NOT assume presence of flagella unless clearly visible.
- Distinguish between:
  - flagellated protozoa (undulating membrane, kinetoplast)
  - helminth larvae (microfilaria: sheath, tapered tail, nuclear column)
- If sheath or nuclear pattern cannot be confirmed, state uncertainty explicitly.

═══════════════════════════════════════════════════════════
STEP 2: INTERPRETATION (match observations to diagnosis)
═══════════════════════════════════════════════════════════

Based ONLY on your observations above, determine if they match any of these patterns. Include species if hallmarks match:

MALARIA: Ring forms, trophozoites, schizonts, or crescents INSIDE RED BLOOD CELLS
- Requires: clear RBCs visible with parasites contained within them
- Ring forms: small rings with chromatin dot(s) inside RBC
- P. falciparum: thin delicate rings, often multiple per RBC, appliqué forms, crescent gametocytes
- P. vivax/ovale: larger rings, Schüffner's dots, enlarged/amoeboid trophozoites, round gametocytes
- P. malariae: band forms, compact schizonts
- Hemozoin pigment often visible in trophozoites

LEISHMANIASIS: Small oval amastigotes (2-4μm) INSIDE MACROPHAGES
- Requires: large cells (macrophages) containing clusters of tiny oval bodies
- Each amastigote has nucleus + kinetoplast ("double dot")
- Found in tissue, bone marrow, NOT typically in peripheral blood smears
- Visceral (L. donovani): in bone marrow/spleen; Cutaneous (L. major/tropica): in skin

TRYPANOSOMIASIS: Elongated flagellates FREE IN PLASMA (extracellular)
- Requires: serpentine organisms BETWEEN cells, not inside them
- Has undulating membrane and free flagellum
- Size: 15-30μm, clearly larger than RBCs
- African (T. brucei): slender, in blood/CSF; American (T. cruzi): C-shaped, broader kinetoplast

FILARIASIS: Long thin larvae (microfilariae) FREE IN BLOOD
- Requires: very long worm-like structures (200-300μm)
- May be sheathed; NO flagellum
- Wuchereria bancrofti: sheathed, no nuclei in tail tip
- Brugia malayi: sheathed, two nuclei in tail tip

SCHISTOSOMIASIS: Oval eggs with spines in urine/stool/tissue
- Requires: large eggs (100-150μm) with lateral (S. mansoni) or terminal (S. haematobium) spine
- Eggs may contain miracidium; often in clusters with inflammatory response

ONCHOCERCIASIS: Unsheathed microfilariae in skin snips
- Requires: short (220-360μm) unsheathed larvae in tissue fluid
- No sheath; nuclei extend to blunt tail

LOIASIS: Sheathed microfilariae in blood
- Requires: medium (230-250μm) sheathed larvae
- Nuclei continuous to pointed tail

═══════════════════════════════════════════════════════════
STEP 3: DIAGNOSIS CHECKLIST
═══════════════════════════════════════════════════════════

Before choosing a diagnosis, confirm ALL required hallmarks:

Malaria:
- Parasite INSIDE RBC
- Ring/trophozoite/schizont/gametocyte identified
- Hemozoin pigment OR chromatin dot seen
- Species: Match specific features (e.g., multiple rings for P. falciparum)

Leishmaniasis:
- Parasites INSIDE macrophages
- Size 2–4 μm
- Nucleus + kinetoplast visible
- Species: Based on sample (visceral vs cutaneous)

Trypanosomiasis:
- Extracellular organism
- Undulating membrane
- Free flagellum
- Species: Slender (African) vs C-shaped (American)

Filariasis:
- Extracellular long larvae
- Size >200μm
- Sheath presence/absence
- Species: Tail nuclei pattern

Schistosomiasis:
- Eggs with distinct spine
- Size 100-150μm
- Species: Spine position (lateral/terminal)

Onchocerciasis:
- Unsheathed microfilariae
- In skin/tissue
- Blunt tail with nuclei

Loiasis:
- Sheathed microfilariae
- Pointed tail with continuous nuclei

If ANY box cannot be confidently checked → choose "Unclear". For "Negative", confirm: 200+ HPFs examined, specific parasites ruled out, optimal staining/focus.

═══════════════════════════════════════════════════════════
STEP 4: FINAL DIAGNOSIS
═══════════════════════════════════════════════════════════

Choose ONE: Malaria, Leishmaniasis, Schistosomiasis, Filariasis, Trypanosomiasis, Onchocerciasis, Loiasis, Negative for Parasites, Unclear

SEVERITY RULE:
Base severity ONLY on organism count per field.
Do NOT guess severity without numeric estimation.

Severity guidance:
- Scanty (+): 1–10 parasites per 100 HPF
- Moderate (++): 1–10 parasites per 10 HPF  
- Heavy (+++): >10 parasites per HPF or macrophages packed with organisms

RULES:
- Your diagnosis MUST be supported by your Step 1 observations
- If you described organisms inside macrophages → cannot be Malaria
- If you described organisms inside RBCs → cannot be Leishmaniasis or Trypanosomiasis
- If sample is tissue/biopsy and you see organisms in large cells → likely Leishmaniasis
- If you cannot see clear parasites, choose "Unclear" (not "Negative" unless HIGH confidence)
- To report "Negative for Parasites", you MUST state:
  • Number of HPFs examined
  • What parasites were specifically looked for
  • Adequate staining and focus confirmed
  If any of the above are missing → use "Unclear"

Return as JSON:
{{
  "observed_background": "<what cells/structures form the background>",
  "observed_organisms": "<description of any organisms seen, or 'None identified'>",
  "organism_location": "<inside RBCs | inside macrophages | extracellular | none seen>",
  "detected_disease": "<diagnosis>",
  "severity": "Scanty (+)|Moderate (++)|Heavy (+++)|N/A",
  "morphology_proof": "<specific features that support your diagnosis>",
  "confidence": "High|Medium|Low",
  "findings": "<comprehensive description>",
  "recommendation": "<next steps>",
  "species": "<species if identifiable, else Unknown>"
}}"""

    return prompt

def get_sample_specific_guidance(sample_type):
    """Return sample-type specific observation guidance"""
    
    sample_lower = sample_type.lower()
    
    if "tissue" in sample_lower or "biopsy" in sample_lower or "skin" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Tissue/Biopsy):
• This is a TISSUE section - you will see tissue architecture, NOT free-flowing blood
• Look for: macrophages, histiocytes, inflammatory cells, tissue structure
• Parasites here are typically INSIDE tissue macrophages (Leishmaniasis) or in tissue spaces
• You should NOT see free RBCs floating as in a blood smear
• Malaria ring forms would NOT be expected in tissue sections
"""
    
    elif "bone marrow" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Bone Marrow):
• Bone marrow contains: hematopoietic cells, megakaryocytes, fat cells, macrophages
• Key finding: Look for macrophages containing intracellular organisms (Leishmaniasis)
• Leishmaniasis amastigotes appear as small oval bodies clustered inside large macrophages
• While RBCs are present, malaria is rarely diagnosed on marrow (use peripheral blood)
• If you see small organisms inside MACROPHAGES → think Leishmaniasis
"""
    
    elif "lymph" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Lymph Node):
• Contains lymphoid tissue with macrophages in sinuses
• Leishmaniasis: look for amastigotes inside macrophages
• Trypanosomiasis: may see trypomastigotes in aspirate fluid
"""
    
    elif "blood" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Blood Smear):
• Background should show RBCs (pink/salmon colored biconcave discs)
• MALARIA: Look for parasites INSIDE RBCs - rings, trophozoites, schizonts, gametocytes
• TRYPANOSOMIASIS: Look for elongated flagellates BETWEEN RBCs (extracellular)
• FILARIASIS: Look for very long thin worm-like larvae in plasma
• Pay attention to RBC size (enlarged in P. vivax/ovale)
"""
    
    elif "csf" in sample_lower or "cerebrospinal" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (CSF):
• Normally acellular or few lymphocytes
• Trypanosomiasis: may see motile trypomastigotes
• Low cellularity - careful examination needed
"""
    
    elif "stool" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Stool):
• Look for: helminth eggs, larvae, protozoan cysts/trophozoites
• Schistosomiasis: eggs with lateral or terminal spine
• Background: fecal debris, bacteria, food particles
"""
    
    elif "urine" in sample_lower:
        return """
SAMPLE-SPECIFIC GUIDANCE (Urine):
• Schistosoma haematobium: eggs with terminal spine
• Background: epithelial cells, crystals, possibly RBCs
"""
    
    else:
        return """
SAMPLE-SPECIFIC GUIDANCE:
• Carefully examine the background to understand the sample type
• Note what cell types are visible before looking for parasites
"""

# ==========================================
# 7. INFERENCE
# ==========================================

def apply_morphology_guardrails(res: ClinicalAnalysis, sample_type: str = "") -> ClinicalAnalysis:
    """
    Apply morphology-based guardrails including sample-type validation.
    Catches biologically impossible combinations.
    """
    # =========================
    # FILARIA vs TRYPANOSOME - MOST IMPORTANT FIX
    # =========================
    if sample_type.lower().startswith("blood smear"):
        disease = res.detected_disease.lower()
        findings = (res.findings + " " + res.morphology_proof).lower()
        
        if "microfilaria" in findings or "sheathed" in findings:
            res.detected_disease = "Filariasis"
            res.species = "Wuchereria bancrofti"
            
        elif any(x in findings for x in ["undulating membrane", "free flagellum", "kinetoplast"]):
            res.detected_disease = "Trypanosomiasis"
            
        else:
            # Ambiguous extracellular worm
            res.confidence = "Moderate"
            res.recommendation += (
                " Morphology is ambiguous between microfilaria and trypanosome; "
                "evaluate sheath, nuclear pattern, and tail morphology."
            )
    
    # =========================
    # SAMPLE-TYPE PRIORITY RULE (Critical for Thick Smear)
    # =========================
    if "thick" in sample_type.lower():
        if res.detected_disease == "Trypanosomiasis":
            res.confidence = "Moderate"
            res.recommendation += (
                " Thick blood smears are more commonly used for microfilariae detection; "
                "consider Filariasis if sheath or nuclear column is identified."
            )
    
    t = (res.morphology_proof + " " + res.findings + " " + 
         res.observed_organisms + " " + res.observed_background).lower()
    
    sample_lower = sample_type.lower()
    is_tissue_sample = any(x in sample_lower for x in ["tissue", "biopsy", "skin snip", "bone marrow", "lymph"])
    is_blood_sample = any(x in sample_lower for x in ["blood", "smear"])
    is_excreta_sample = any(x in sample_lower for x in ["stool", "urine"])
    is_csf_sample = "csf" in sample_lower
    
    # === Location indicators ===
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
    
    # === Morphology indicators ===
    # Malaria
    says_ring = any(p in t for p in ["ring form", "ring-form", "signet ring", "delicate ring", "rings with"])
    says_crescent = any(p in t for p in ["crescent", "banana", "banana-shaped", "crescentic"])
    says_schizont = "schizont" in t or "merozoite" in t
    says_trophozoite = "trophozoite" in t
    says_gametocyte = "gametocyte" in t
    malaria_indicators = says_ring or says_crescent or says_schizont or says_gametocyte
    
    # Leishmania
    says_amastigote = any(p in t for p in ["amastigote", "ld bod", "leishman-donovan", "oval bod"])
    says_amastigote_size = any(p in t for p in ["2-4 μm", "2–4 μm", "tiny", "much smaller than rbc"])
    says_small_oval_in_macro = says_in_macrophage and says_amastigote_size
    leishmania_indicators = says_amastigote or says_small_oval_in_macro
    
    # Trypanosome
    says_flagellum = "flagell" in t
    says_undulating = "undulating membrane" in t
    says_trypomastigote = "trypomastigote" in t
    tryp_indicators = says_flagellum or says_undulating or says_trypomastigote
    
    # Microfilaria (general)
    says_microfilaria = "microfilar" in t
    says_sheathed = "sheathed" in t
    says_unsheathed = any(p in t for p in ["unsheathed", "no sheath"])
    says_larval = any(p in t for p in ["larva", "worm-like", "long thin"])
    microfilaria_indicators = says_microfilaria or (says_larval and says_extracellular)
    
    # Schistosoma
    says_egg = "egg" in t
    says_spine = any(p in t for p in ["spine", "terminal spine", "lateral spine"])
    says_miracidium = "miracidium" in t
    schisto_indicators = says_egg and says_spine
    
    # Species-specific
    says_multiple_rings = any(p in t for p in ["multiple rings", "multiple per rbc", "appliqué"])
    says_schuffner = "schüffner" in t or "schuffner" in t
    says_band_form = "band form" in t
    says_c_shaped = "c-shaped" in t or "c shaped" in t
    says_tail_nuclei = any(p in t for p in ["tail nuclei", "nuclei in tail"])
    says_blunt_tail = "blunt tail" in t
    says_pointed_tail = "pointed tail" in t
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 1: SAMPLE-TYPE IMPOSSIBILITIES
    # ═══════════════════════════════════════════════════════════════
    
    # Tissue samples should NOT yield malaria diagnosed on "RBC ring forms"
    if is_tissue_sample and res.detected_disease == "Malaria" and says_in_rbc and not is_blood_sample:
        # Check if this is actually describing amastigotes in macrophages
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
    
    # Bone marrow with "ring forms" is likely Leishmania amastigotes misidentified
    if "bone marrow" in sample_lower and res.detected_disease == "Malaria":
        if says_in_macrophage or "macrophage" in t or "histiocyte" in t:
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis", 
                "confidence": "Medium",
                "species": "Leishmania donovani",  # Common in visceral
                "recommendation": "Bone marrow with intracellular organisms in macrophages is classic for Visceral Leishmaniasis. Confirm with rK39 serology or PCR.",
                "morphology_proof": "Amastigotes identified within bone marrow macrophages."
            })
        # Even without explicit macrophage mention, bone marrow + small intracellular = likely Leishmania
        if any(p in t for p in ["small", "oval", "round bodies", "clusters"]):
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium", 
                "species": "Leishmania spp.",
                "recommendation": "Bone marrow aspirate with small intracellular organisms suggests Visceral Leishmaniasis. Peripheral blood is preferred for malaria diagnosis.",
            })
    
    # Schistosomiasis not in blood/tissue without eggs
    if res.detected_disease == "Schistosomiasis" and not is_excreta_sample and not schisto_indicators:
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Schistosomiasis requires eggs in urine/stool. Re-evaluate for microfilariae if worm-like."
        })
    
    # Onchocerciasis only in skin snips
    if res.detected_disease == "Onchocerciasis" and not "skin" in sample_lower:
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Onchocerciasis microfilariae are in skin snips. Check sample type."
        })
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 2: LEISHMANIASIS POSITIVE IDENTIFICATION
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease != "Leishmaniasis" and leishmania_indicators:
        if is_tissue_sample or says_in_macrophage:
            species = "Leishmania donovani" if any(x in sample_lower for x in ["bone", "spleen"]) else "Leishmania spp."
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Amastigotes in macrophages/tissue indicate Leishmaniasis. Speciate with PCR. Assess for visceral involvement if bone marrow/spleen positive."
            })
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 3: MALARIA VALIDATION (only on blood samples)
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Malaria":
        # Malaria should only be diagnosed on blood samples
        if is_tissue_sample and not is_blood_sample:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": f"Malaria diagnosis on {sample_type} is unusual. Use peripheral blood smear for malaria diagnosis."
            })
        
        # Malaria must show intraerythrocytic parasites
        if not (says_in_rbc or malaria_indicators):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Malaria diagnosis requires identification of intraerythrocytic parasites (rings, trophozoites, schizonts, or gametocytes)."
            })
        
        # Species correction
        if res.species == "Unknown":
            if says_multiple_rings or says_crescent:
                res.species = "P. falciparum"
            elif says_schuffner or says_band_form:
                res.species = "P. vivax" if says_schuffner else "P. malariae"
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 4: TRYPANOSOMIASIS VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Trypanosomiasis":
        # Cannot be inside RBCs
        if says_in_rbc and not says_extracellular:
            if malaria_indicators or says_in_rbc:
                return res.model_copy(update={
                    "detected_disease": "Unclear",
                    "confidence": "Low",
                    "recommendation": "Trypanosomes are extracellular. Organisms inside RBCs suggest Malaria instead."
                })
        
        # Cannot be inside macrophages (that's Leishmania)
        if says_in_macrophage and not says_extracellular:
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Organisms inside macrophages indicate Leishmaniasis, not Trypanosomiasis."
            })
        
        # Species correction
        if says_c_shaped:
            res.species = "Trypanosoma cruzi"
        elif is_csf_sample:
            res.species = "Trypanosoma brucei"
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 5: FILARIASIS VALIDATION  
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Filariasis":
        if says_in_rbc or (says_flagellum and not says_microfilaria):
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Microfilariae are extracellular larvae without flagella. Re-examine for correct identification."
            })
        
        # Species correction
        if says_sheathed and says_tail_nuclei:
            res.species = "Brugia malayi"
        elif says_sheathed:
            res.species = "Wuchereria bancrofti"
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 5.1: SCHISTOSOMIASIS VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Schistosomiasis":
        if not schisto_indicators:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Schistosomiasis requires eggs with spines. Check for other helminths."
            })
        
        # Species correction
        if "terminal spine" in t:
            res.species = "Schistosoma haematobium"
        elif "lateral spine" in t:
            res.species = "Schistosoma mansoni"
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 5.2: ONCHOCERCIASIS / LOIASIS VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
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
        
        # Promote unclear to specific if indicators match
        if res.detected_disease == "Unclear" and microfilaria_indicators:
            if says_unsheathed and says_blunt_tail and "skin" in sample_lower:
                return res.model_copy(update={
                    "detected_disease": "Onchocerciasis",
                    "confidence": "Medium",
                    "species": "Onchocerca volvulus",
                    "recommendation": "Unsheathed microfilariae in skin suggest Onchocerciasis."
                })
            elif says_sheathed and says_pointed_tail and is_blood_sample:
                return res.model_copy(update={
                    "detected_disease": "Loiasis",
                    "confidence": "Medium",
                    "species": "Loa loa",
                    "recommendation": "Sheathed microfilariae with pointed tail suggest Loiasis."
                })
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 6: CONSERVATIVE UNCLEAR PROMOTION
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Unclear":
        # Promote to Trypanosomiasis if clearly extracellular + flagellated
        if tryp_indicators and says_extracellular and not says_in_rbc and not says_in_macrophage:
            if is_blood_sample or is_csf_sample and not leishmania_indicators:
                species = "Trypanosoma cruzi" if says_c_shaped else "Trypanosoma brucei"
                return res.model_copy(update={
                    "detected_disease": "Trypanosomiasis",
                    "confidence": "Medium",
                    "species": species,
                    "recommendation": "Extracellular flagellated organisms suggest Trypanosomiasis. Confirm species with concentration techniques."
                })
        
        # Promote to Leishmaniasis if tissue + macrophage organisms
        if is_tissue_sample and (says_in_macrophage or says_amastigote):
            return res.model_copy(update={
                "detected_disease": "Leishmaniasis",
                "confidence": "Medium",
                "species": "Leishmania spp.",
                "recommendation": "Intracellular organisms in tissue macrophages suggest Leishmaniasis."
            })
        
        # Promote to Schistosomiasis if eggs with spines
        if schisto_indicators and is_excreta_sample:
            species = "Schistosoma haematobium" if "terminal spine" in t else "Schistosoma mansoni" if "lateral spine" in t else "Schistosoma spp."
            return res.model_copy(update={
                "detected_disease": "Schistosomiasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Eggs with spines in excreta suggest Schistosomiasis. Confirm with Kato-Katz."
            })
        
        # Promote to Filariasis if general microfilariae
        if microfilaria_indicators and says_sheathed and is_blood_sample:
            species = "Brugia malayi" if says_tail_nuclei else "Wuchereria bancrofti"
            return res.model_copy(update={
                "detected_disease": "Filariasis",
                "confidence": "Medium",
                "species": species,
                "recommendation": "Sheathed microfilariae in blood suggest lymphatic Filariasis."
            })
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 7: NEGATIVE VALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    if res.detected_disease == "Negative for Parasites":
        if res.confidence != "High":
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "severity": "N/A",
                "recommendation": "Cannot confidently exclude parasites. Examine additional fields."
            })
        # Check for contradictions - if any organisms mentioned, can't be negative
        if any(p in t for p in ["organism", "parasite", "seen", "observed", "identified", "present"]) and "no " not in t and "none" not in t:
            return res.model_copy(update={
                "detected_disease": "Unclear",
                "confidence": "Low",
                "recommendation": "Findings mention structures but diagnosis is negative. Manual review required."
            })
        # Enforce stricter negative criteria: 200+ HPFs
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
    
    # ═══════════════════════════════════════════════════════════════
    # RULE 8: SIZE-AWARE GUARDRAIL (RBC vs Amastigote confusion)
    # ═══════════════════════════════════════════════════════════════
    
    # RBC-sized structures (~7 μm) CANNOT be amastigotes
    if says_in_macrophage and any(p in t for p in ["rbc-sized", "same size as rbc", "7 μm", "7um", "similar to rbc"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "RBC-sized structures inside macrophages cannot be amastigotes (2-4 μm). Re-evaluate identification.",
            "morphology_proof": "Size mismatch: described structures are RBC-sized, but amastigotes are much smaller (2-4 μm)."
        })
    
    # Egg size guardrail for Schisto
    if schisto_indicators and any(p in t for p in ["small egg", "<50 μm", "tiny egg"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Schistosome eggs are 100-150μm. Small eggs may indicate other helminths."
        })
    
    # Microfilariae size guardrail
    if microfilaria_indicators and any(p in t for p in ["short larva", "<100 μm", "tiny worm"]):
        return res.model_copy(update={
            "detected_disease": "Unclear",
            "confidence": "Low",
            "recommendation": "Microfilariae are >200μm. Re-evaluate for protozoa if smaller."
        })
    
    return res

def run_agent(image, sample_type, magnification, stain, patient_context, processor, model, use_enhancement=True):
    """Run the MedGemma inference pipeline"""
    st.session_state.debug_log = []
    
    # Image enhancement
    if use_enhancement:
        processed_image = enhance_image(image, sample_type)
        log_debug("IMAGE_ENHANCED", f"Applied enhancement for {sample_type}")
    else:
        processed_image = image
        log_debug("IMAGE_ORIGINAL", "Using original image without enhancement")
    
    # Build minimal prompt
    prompt = build_minimal_prompt(sample_type, magnification, stain, patient_context)
    log_debug("PROMPT_LENGTH", f"{len(prompt)} characters")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": processed_image},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        # Prepare inputs
        text_input = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=text_input,
            images=[processed_image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        log_debug("INPUT_TOKENS", f"{inputs['input_ids'].shape[1]} tokens")
        
        if 'pixel_values' in inputs:
            log_debug("IMAGE_PROCESSED", f"Shape: {inputs['pixel_values'].shape}")
        else:
            log_debug("IMAGE_WARNING", "No pixel_values in inputs!")

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0,
            )

        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        decoded = processor.decode(generated, skip_special_tokens=True).strip()
        log_debug("RAW_OUTPUT", decoded[:1000])

    except Exception as e:
        log_debug("INFERENCE_ERROR", traceback.format_exc())
        return ClinicalAnalysis(
            detected_disease="Unclear",
            severity="N/A",
            morphology_proof="Model inference failed",
            confidence="Low",
            findings=f"Error: {str(e)[:200]}",
            recommendation="Please try again or perform manual review",
            species="Unknown"
        ), "ERROR"

    # Parse JSON response
    try:
        clean = re.sub(r"```json|```", "", decoded).strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*"detected_disease"[\s\S]*?\}', clean)
        
        if json_match:
            json_str = json_match.group()
            # Parse and validate
            import json
            parsed = json.loads(json_str)
            
            # Ensure required fields exist with defaults
            parsed.setdefault("observed_background", "")
            parsed.setdefault("observed_organisms", "")  
            parsed.setdefault("organism_location", "")
            parsed.setdefault("species", "Unknown")
            parsed.setdefault("severity", "N/A")
            
            result = ClinicalAnalysis(**parsed)

            # Apply morphology guardrails
            result = apply_morphology_guardrails(
                result,
                sample_type=sample_type
            )
            
            # =========================
            # SPECIES INFERENCE RULE
            # =========================
            # Species only resolved when morphology + context align
            if result.confidence != "High" or "consistent with" not in result.morphology_proof.lower():
                result.species = "Unknown"

            return result, decoded

        else:
            return ClinicalAnalysis(
                detected_disease="Unclear",
                severity="N/A",
                morphology_proof="Model output could not be parsed as valid JSON.",
                confidence="Low",
                findings=decoded[:500],
                recommendation="Model response was not structured correctly. Review raw output.",
                species="Unknown"
            ), decoded

    except Exception as e:
        log_debug("JSON_PARSE_ERROR", traceback.format_exc())
        return ClinicalAnalysis(
            detected_disease="Unclear",
            severity="N/A",
            morphology_proof="Failed to parse model output",
            confidence="Low",
            findings=f"Parsing error: {str(e)[:200]}",
            recommendation="Review raw output and validate model response manually.",
            species="Unknown"
        ), decoded

# ==========================================
# 8. MAIN UI
# ==========================================
def main():
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("🦟 NTD-Assist")
        st.caption("AI-Powered NTD Detection")
        st.markdown("---")
        
        st.subheader("📋 Sample Settings")
        
        sample = st.selectbox(
            "Sample Type", 
            [
                "Blood Smear (Thin)",
                "Blood Smear (Thick)", 
                "Tissue Biopsy",
                "Skin Snip",
                "Bone Marrow Aspirate",
                "Lymph Node Aspirate", 
                "Urine Sediment", 
                "Stool Sample",
                "CSF (Cerebrospinal Fluid)",
                "Other/Unknown"
            ],
            help="Select the specimen type being examined"
        )
        
        mag = st.selectbox(
            "Magnification", 
            [
                "1000x (Oil Immersion)",
                "400x (High Dry)", 
                "100x (Low Power)",
                "40x (Scanning)",
                "Unknown"
            ],
            help="Microscope magnification used"
        )
        
        stain = st.selectbox(
            "Stain", 
            [
                "Giemsa",
                "Wright-Giemsa",
                "Wright",
                "Field's Stain",
                "H&E (Hematoxylin & Eosin)",
                "Iodine/Lugol's",
                "Modified Acid-Fast",
                "Trichrome",
                "Unstained (Wet Mount)",
                "Unstained (Dry)",
                "Other/Unknown"
            ],
            help="Staining method used on the sample"
        )
        
        st.markdown("---")
        st.subheader("⚙️ Options")
        
        use_enhancement = st.checkbox(
            "🔧 Image Enhancement", 
            value=True,
            help="Apply contrast/color enhancement to improve parasite visibility"
        )
        
        show_enhanced = st.checkbox(
            "👁️ Show Enhanced Preview",
            value=True,
            help="Display original and enhanced images side by side"
        )
        
        st.markdown("---")
        model_status = st.empty()
        
        # Info section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **NTD-Assist** uses Google's MedGemma model to analyze microscopy images for Neglected Tropical Diseases.
            
            **Supported Diseases:**
            - Malaria (Plasmodium spp.)
            - Leishmaniasis
            - Trypanosomiasis
            - Filariasis
            - Schistosomiasis
            - Onchocerciasis
            - Loiasis
            
            **⚠️ Disclaimer:** For educational/screening purposes only. Always confirm with qualified professionals.
            """)
    
    # ===== LOAD MODEL =====
    model_status.info("⏳ Loading model...")
    
    with st.spinner("🔄 Loading MedGemma (1-2 min on first run)..."):
        processor, model, error = load_medgemma()
    
    if error:
        model_status.error("❌ Model failed")
        st.error(f"**Model Loading Error:**\n```\n{error}\n```")
        st.info("**Troubleshooting:**\n"
                "1. Verify HF_TOKEN has MedGemma access\n"
                "2. Check GPU memory availability\n"
                "3. Try restarting the kernel")
        return
    
    model_status.success("✅ Model Ready")
    
    # ===== MAIN CONTENT =====
    st.header("🔬 NTD-Assist | Microscopy Analysis")
    st.markdown("Upload a microscopy image to detect parasites causing Neglected Tropical Diseases")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        uploaded = st.file_uploader(
            "Upload Microscopy Image", 
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Supported formats: PNG, JPG, JPEG, TIF, BMP"
        )
        
        context = st.text_area(
            "Patient Context (Optional)",
            placeholder="e.g., 35yo male, fever for 5 days, returned from Nigeria 2 weeks ago, hepatosplenomegaly...",
            height=100,
            help="Clinical information helps improve diagnostic accuracy"
        )
    
    with col2:
        if uploaded:
            # Load image
            img = Image.open(uploaded).convert("RGB")
            
            # Display images
            if show_enhanced and use_enhancement:
                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.image(img, caption="📷 Original", use_container_width=True)
                with col_enh:
                    enhanced_preview = enhance_image(img, sample)
                    st.image(enhanced_preview, caption="✨ Enhanced", use_container_width=True)
            else:
                st.image(img, caption=f"📷 {uploaded.name}", use_container_width=True)
            
            # Quality warnings
            warnings = check_image_quality(img)
            if warnings:
                st.warning(f"⚠️ Image Quality Issues: {', '.join(warnings)}")
            
            # Analysis button
            st.markdown("---")
            
            if st.button("🔬 Analyze Slide", type="primary", use_container_width=True):
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Preprocessing image...")
                progress_bar.progress(20)
                
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(40)
                
                start_time = time.time()
                
                # Run inference
                res, status = run_agent(
                    img, sample, mag, stain, context, 
                    processor, model, use_enhancement
                )
                
                elapsed = time.time() - start_time
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Color-coded diagnosis banner
                if "Negative" in res.detected_disease:
                    st.success(f"## ✅ {res.detected_disease}")
                elif res.detected_disease == "Unclear":
                    st.warning(f"## ⚠️ {res.detected_disease}")
                else:
                    st.error(f"## 🚨 {res.detected_disease}")
                
                # Key metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Species", res.species)
                with metric_col2:
                    st.metric("Severity", res.severity)
                with metric_col3:
                    st.metric("Confidence", res.confidence)
                
                # Detailed report
                with st.expander("📋 Detailed Report", expanded=True):
                    st.markdown("**Morphological Evidence:**")
                    st.info(res.morphology_proof)
                    
                    st.markdown("**Detailed Findings:**")
                    st.write(res.findings)
                    
                    st.markdown("**Recommendation:**")
                    st.success(res.recommendation)
                
                # Download PDF
                if res.detected_disease not in ["Unclear"]:
                    st.markdown("---")
                    pdf_bytes = create_pdf(res, sample, stain, mag, context)
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"NTD_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                # Status footer
                st.caption(f"⏱️ Analysis completed in {elapsed:.1f}s | Status: {status}")
                
        else:
            # No image uploaded - show instructions
            st.info("👆 Upload a microscopy image to begin analysis")
            
            with st.expander("📖 Quick Reference Guide", expanded=False):
                st.markdown("""
### Supported Diseases & Optimal Conditions

| Disease | Sample | Stain | Key Features |
|---------|--------|-------|--------------|
| **Malaria** | Blood (thin/thick) | Giemsa | Ring forms in RBCs |
| **Leishmaniasis** | Tissue, Bone marrow | Giemsa, H&E | Amastigotes in macrophages |
| **Trypanosomiasis** | Blood, CSF | Giemsa | Trypomastigotes (extracellular) |
| **Filariasis** | Blood (night) | Giemsa | Sheathed microfilariae |
| **Schistosomiasis** | Urine, Stool | Unstained | Eggs with spines |
| **Onchocerciasis** | Skin snip | Giemsa | Unsheathed microfilariae |

### Magnification Guide

| Mag | Best For |
|-----|----------|
| **1000x** | Malaria species ID, Leishmania, Trypanosomes |
| **400x** | Microfilariae, Helminth eggs |
| **100x** | Large eggs, Screening |

### Tips for Best Results
1. Use well-stained, properly focused images
2. Capture at appropriate magnification for the target
3. Include clinical context for better species prediction
4. Multiple images may help confirm findings
                """)
    
    # ===== DEBUG SECTION =====
    st.markdown("---")
    if st.checkbox("🐛 Show Debug Log"):
        if st.session_state.debug_log:
            for entry in st.session_state.debug_log:
                with st.expander(f"{entry['stage']} - {entry['timestamp'].split('T')[1][:8]}"):
                    st.code(entry['data'], language="json")
        else:
            st.info("No debug data yet. Run an analysis first.")

if __name__ == "__main__":
    main()