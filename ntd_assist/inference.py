import datetime
import json
import re
import time
import traceback

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from . import config
from .guardrails import apply_morphology_guardrails
from .image_processing import enhance_image
from .prompt import build_minimal_prompt
from .schema import ClinicalAnalysis


def load_cpu_model():
    """Load Qwen2-VL-2B-Instruct on CPU for HF Spaces / no-GPU environments"""
    try:
        min_pixels = 256 * 28 * 28
        max_pixels = 640 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            config.CPU_MODEL_ID,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            config.CPU_MODEL_ID,
            device_map={"": "cpu"},
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        return processor, model, None
    except Exception as e:
        return None, None, str(e)


def load_model():
    """Auto-select model: MedGemma if GPU available, Qwen2-VL on CPU otherwise"""
    if config.USE_CPU_MODEL:
        return load_cpu_model()
    if torch.cuda.is_available():
        return load_medgemma()
    return load_cpu_model()


def load_medgemma():
    has_cuda = torch.cuda.is_available()
    try:
        processor = AutoProcessor.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        if has_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForImageTextToText.from_pretrained(
                config.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                config.MODEL_ID,
                device_map={"": "cpu"},
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        return processor, model, None
    except Exception as e:
        return None, None, str(e)


def run_agent(image, sample_type, magnification, stain, patient_context, processor, model,
              use_enhancement=True, log=None):
    """Run the MedGemma inference pipeline"""
    log = log or (lambda *a, **k: None)
    log("RUN_START", f"---- {datetime.datetime.now().isoformat()} ----")

    if use_enhancement:
        processed_image = enhance_image(image, sample_type, log=log)
        log("IMAGE_ENHANCED", f"Applied enhancement for {sample_type}")
    else:
        processed_image = image
        log("IMAGE_ORIGINAL", "Using original image without enhancement")

    prompt = build_minimal_prompt(sample_type, magnification, stain, patient_context)
    log("PROMPT_LENGTH", f"{len(prompt)} characters")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": processed_image},
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=text_input,
            images=[processed_image],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        log("INPUT_TOKENS", f"{inputs['input_ids'].shape[1]} tokens")

        if 'pixel_values' in inputs:
            log("IMAGE_PROCESSED", f"Shape: {inputs['pixel_values'].shape}")
        else:
            log("IMAGE_WARNING", "No pixel_values in inputs!")

        def _generate(max_new_tokens):
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

        try:
            output = _generate(config.MAX_NEW_TOKENS)
        except torch.cuda.OutOfMemoryError:
            log("OOM_RETRY", "CUDA OOM; clearing cache and retrying with fewer tokens")
            torch.cuda.empty_cache()
            output = _generate(config.MAX_NEW_TOKENS_OOM_RETRY)
        except Exception as gen_err:
            log("GEN_RETRY", f"Transient error, retrying once: {gen_err}")
            time.sleep(config.GEN_RETRY_SLEEP_SEC)
            output = _generate(config.MAX_NEW_TOKENS)

        input_len = inputs["input_ids"].shape[1]
        generated = output[0][input_len:]
        decoded = processor.decode(generated, skip_special_tokens=True).strip()
        log("RAW_OUTPUT", decoded[:1000])

    except Exception as e:
        log("INFERENCE_ERROR", traceback.format_exc())
        return ClinicalAnalysis(
            detected_disease="Unclear",
            severity="N/A",
            morphology_proof="Model inference failed",
            confidence="Low",
            findings=f"Error: {str(e)[:200]}",
            recommendation="Please try again or perform manual review",
            species="Unknown"
        ), "ERROR"

    try:
        clean = re.sub(r"```json|```", "", decoded).strip()
        json_match = re.search(r'\{[\s\S]*"detected_disease"[\s\S]*?\}', clean)

        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)

            parsed.setdefault("observed_background", "")
            parsed.setdefault("observed_organisms", "")
            parsed.setdefault("organism_location", "")
            parsed.setdefault("species", "Unknown")
            parsed.setdefault("severity", "N/A")

            allowed = set(ClinicalAnalysis.model_fields.keys())
            parsed = {k: v for k, v in parsed.items() if k in allowed}

            result = ClinicalAnalysis(**parsed)
            result = apply_morphology_guardrails(result, sample_type=sample_type)
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
        log("JSON_PARSE_ERROR", traceback.format_exc())
        return ClinicalAnalysis(
            detected_disease="Unclear",
            severity="N/A",
            morphology_proof="Failed to parse model output",
            confidence="Low",
            findings=f"Parsing error: {str(e)[:200]}",
            recommendation="Review raw output and validate model response manually.",
            species="Unknown"
        ), decoded
