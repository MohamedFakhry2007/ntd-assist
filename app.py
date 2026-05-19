import datetime
import time

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from ntd_assist import config
from ntd_assist.image_processing import enhance_image, check_image_quality
from ntd_assist.inference import load_model, run_agent
from ntd_assist.pdf_report import create_pdf


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
        "data": str(data)[:config.DEBUG_DATA_TRUNCATE]
    })
    if len(st.session_state.debug_log) > config.DEBUG_LOG_MAX:
        st.session_state.debug_log = st.session_state.debug_log[-config.DEBUG_LOG_MAX:]


@st.cache_resource(show_spinner=False)
def _cached_load_model():
    return load_model()


def main():
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

    model_status.info("⏳ Loading model...")

    with st.spinner("🔄 Loading model (1-2 min on first run)..."):
        processor, model, error = _cached_load_model()

    if error:
        model_status.error("❌ Model failed")
        st.error(f"**Model Loading Error:**\n```\n{error}\n```")
        st.info("**Troubleshooting:**\n"
                "1. Verify HF_TOKEN has model access (if using a gated model)\n"
                "2. Check GPU memory availability\n"
                "3. Try restarting the kernel")
        return

    model_status.success("✅ Model Ready")
    if not torch.cuda.is_available():
        st.sidebar.info("ℹ️ Running on CPU (slow — GPU not detected)")

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
            if getattr(uploaded, "size", 0) > config.MAX_UPLOAD_BYTES:
                st.error(f"Image is {uploaded.size / 1024 / 1024:.1f} MB — limit is "
                         f"{config.MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB. Please downsample and retry.")
                st.stop()

            try:
                img = Image.open(uploaded).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                st.error(f"Could not read image file: {e}")
                st.stop()

            if img.size[0] * img.size[1] > config.MAX_PIXELS:
                img.thumbnail(config.THUMBNAIL_MAX, Image.LANCZOS)
                st.info(f"Image downscaled to {config.THUMBNAIL_MAX[0]}px for processing")

            if show_enhanced and use_enhancement:
                col_orig, col_enh = st.columns(2)
                with col_orig:
                    st.image(img, caption="📷 Original", use_column_width=True)
                with col_enh:
                    enhanced_preview = enhance_image(img, sample, log=log_debug)
                    st.image(enhanced_preview, caption="✨ Enhanced", use_column_width=True)
            else:
                st.image(img, caption=f"📷 {uploaded.name}", use_column_width=True)

            warnings = check_image_quality(img, log=log_debug)
            if warnings:
                st.warning(f"⚠️ Image Quality Issues: {', '.join(warnings)}")

            st.markdown("---")

            if st.button("🔬 Analyze Slide", type="primary"):

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔄 Preprocessing image...")
                progress_bar.progress(20)

                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(40)

                start_time = time.time()

                res, status = run_agent(
                    img, sample, mag, stain, context,
                    processor, model, use_enhancement,
                    log=log_debug
                )

                elapsed = time.time() - start_time

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.markdown("---")
                st.subheader("📊 Analysis Results")

                if "Negative" in res.detected_disease:
                    st.success(f"## ✅ {res.detected_disease}")
                elif res.detected_disease == "Unclear":
                    st.warning(f"## ⚠️ {res.detected_disease}")
                else:
                    st.error(f"## 🚨 {res.detected_disease}")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Species", res.species)
                with metric_col2:
                    st.metric("Severity", res.severity)
                with metric_col3:
                    st.metric("Confidence", res.confidence)

                with st.expander("📋 Detailed Report", expanded=True):
                    st.markdown("**Morphological Evidence:**")
                    st.info(res.morphology_proof)

                    st.markdown("**Detailed Findings:**")
                    st.write(res.findings)

                    st.markdown("**Recommendation:**")
                    st.success(res.recommendation)

                if res.detected_disease not in ["Unclear"]:
                    st.markdown("---")
                    pdf_bytes = create_pdf(res, sample, stain, mag, context)

                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"NTD_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )

                st.caption(f"⏱️ Analysis completed in {elapsed:.1f}s | Status: {status}")

        else:
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
