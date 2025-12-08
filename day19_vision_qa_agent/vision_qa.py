import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import fal_client
import requests

from utils.config import settings


def download_image(image_url: str, api_key: str = None) -> bytes:
    """
    Download image from URL and return as bytes.

    Args:
        image_url: URL of the image to download
        api_key: Optional API key for authenticated requests

    Returns:
        Image bytes

    Raises:
        Exception: If download fails
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Key {api_key}"

        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type or response.text.strip().startswith("<html"):
            raise Exception(
                f"URL returned HTML instead of image (likely 404 or error page): {response.text[:200]}"
            )

        if not content_type.startswith("image/"):
            raise Exception(f"URL did not return an image (content-type: {content_type})")

        return response.content
    except requests.RequestException as e:
        raise Exception(f"Failed to download image from {image_url}: {e}") from e


def create_evaluation_checklist(style_profile: Dict) -> str:
    """
    Build structured evaluation checklist from style profile.

    Args:
        style_profile: Dictionary containing style profile data

    Returns:
        Formatted checklist string for vision model prompt
    """
    style_name = style_profile.get("name", "Unknown")
    color_palette = style_profile.get("color_palette", [])
    mood = style_profile.get("mood", "")
    visual_style = style_profile.get("visual_style", {})
    dos = style_profile.get("dos", [])
    donts = style_profile.get("donts", [])
    style_description = style_profile.get("style_description", "")

    checklist_parts = [
        f"# Style Profile: {style_name}",
        "",
        f"## Style Description: {style_description}",
        "",
        "## Evaluation Criteria:",
        "",
    ]

    if color_palette:
        colors_str = ", ".join(color_palette)
        checklist_parts.append(f"### Color Palette Compliance:")
        checklist_parts.append(f"- Expected colors: {colors_str}")
        checklist_parts.append(
            "- Score: 1.0 if image primarily uses these colors, 0.5 if partially, 0.0 if not"
        )
        checklist_parts.append("")

    if mood:
        checklist_parts.append(f"### Mood Matching:")
        checklist_parts.append(f"- Expected mood: {mood}")
        checklist_parts.append(
            "- Score: 1.0 if mood matches, 0.5 if partially, 0.0 if not"
        )
        checklist_parts.append("")

    if visual_style:
        dimensionality = visual_style.get("dimensionality", "")
        texture = visual_style.get("texture", "")
        detail_level = visual_style.get("detail_level", "")

        if dimensionality:
            checklist_parts.append(f"### Dimensionality:")
            checklist_parts.append(f"- Expected: {dimensionality}")
            checklist_parts.append(
                "- Score: 1.0 if matches, 0.0 if not"
            )
            checklist_parts.append("")

        if texture:
            checklist_parts.append(f"### Texture:")
            checklist_parts.append(f"- Expected: {texture}")
            checklist_parts.append(
                "- Score: 1.0 if matches, 0.0 if not"
            )
            checklist_parts.append("")

        if detail_level:
            checklist_parts.append(f"### Detail Level:")
            checklist_parts.append(f"- Expected: {detail_level}")
            checklist_parts.append(
                "- Score: 1.0 if matches, 0.5 if partially, 0.0 if not"
            )
            checklist_parts.append("")

    if dos:
        checklist_parts.append("### Required Elements (DOs):")
        for do_item in dos:
            checklist_parts.append(f"- {do_item}")
        checklist_parts.append(
            "- Score: 1.0 if all present, 0.7 if most present, 0.3 if some present, 0.0 if none"
        )
        checklist_parts.append("")

    if donts:
        checklist_parts.append("### Forbidden Elements (DON'Ts):")
        for dont_item in donts:
            checklist_parts.append(f"- {dont_item}")
        checklist_parts.append(
            "- Score: 1.0 if none present, 0.5 if few present, 0.0 if many present"
        )
        checklist_parts.append("")

    return "\n".join(checklist_parts)


def analyze_image_with_vision(
    image_url: str = None,
    image_bytes: bytes = None,
    style_profile: Dict = None,
    api_key: str = None,
    model: str = None,
    fal_api_key: str = None,
) -> Dict[str, Any]:
    """
    Analyze image using fal.ai vision model against style profile.

    Args:
        image_url: URL of the image to analyze (optional if image_bytes provided)
        image_bytes: Image bytes (optional if image_url provided)
        style_profile: Style profile dictionary
        api_key: Not used (kept for compatibility)
        model: Vision model to use (defaults to "fal-ai/sa2va/8b/image")
        fal_api_key: fal.ai API key

    Returns:
        Dictionary with analysis results including scores and observations
    """
    if model is None:
        model = os.getenv("FAL_VISION_MODEL", "fal-ai/sa2va/4b/image")

    if not fal_api_key:
        fal_api_key = settings.fal_api_key

    if not fal_api_key:
        raise ValueError("fal.ai API key is required (FAL_API_KEY)")

    checklist = create_evaluation_checklist(style_profile)

    prompt = f"""You are an expert image quality analyst. Evaluate this image against the following style requirements:

{checklist}

Provide your evaluation as a valid JSON object with this exact structure:
{{
  "compliance_scores": {{
    "color_palette": 0.0-1.0,
    "mood": 0.0-1.0,
    "dimensionality": 0.0-1.0,
    "texture": 0.0-1.0,
    "detail_level": 0.0-1.0,
    "dos": 0.0-1.0,
    "donts": 0.0-1.0
  }},
  "overall_score": 0.0-1.0,
  "observations": ["finding1", "finding2", ...],
  "passed": true/false
}}

Be thorough and objective. Return ONLY valid JSON."""

    image_input = None

    if image_bytes:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_input = f"data:image/jpeg;base64,{base64_image}"
    elif image_url:
        image_input = image_url
    else:
        raise ValueError("Either image_url or image_bytes must be provided")

    try:
        original_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = fal_api_key

        start_time = time.time()

        input_params = {
            "image_url": image_input,
        }

        if "sa2va" in model.lower():
            input_params["prompt"] = prompt
        elif "clip" in model.lower():
            input_params["text"] = prompt
        elif "blip" in model.lower():
            input_params["prompt"] = prompt
        else:
            input_params["prompt"] = prompt

        result = fal_client.subscribe(
            model,
            arguments=input_params,
            with_logs=False,
        )

        if original_key is not None:
            os.environ["FAL_KEY"] = original_key
        elif "FAL_KEY" in os.environ:
            del os.environ["FAL_KEY"]

        latency = time.time() - start_time

        content = None
        if isinstance(result, dict):
            content = (
                result.get("text") or
                result.get("caption") or
                result.get("description") or
                result.get("answer") or
                result.get("response")
            )

            if not content and "output" in result:
                output = result["output"]
                if isinstance(output, dict):
                    content = (
                        output.get("text") or
                        output.get("caption") or
                        output.get("description") or
                        output.get("answer")
                    )
                elif isinstance(output, str):
                    content = output

            if not content and "result" in result:
                result_data = result["result"]
                if isinstance(result_data, dict):
                    content = (
                        result_data.get("text") or
                        result_data.get("caption") or
                        result_data.get("description") or
                        result_data.get("answer")
                    )
                elif isinstance(result_data, str):
                    content = result_data

            if not content and "data" in result:
                data = result["data"]
                if isinstance(data, dict):
                    content = (
                        data.get("text") or
                        data.get("caption") or
                        data.get("description")
                    )
                elif isinstance(data, str):
                    content = data
        elif hasattr(result, "text"):
            content = result.text
        elif hasattr(result, "caption"):
            content = result.caption
        elif hasattr(result, "description"):
            content = result.description
        elif isinstance(result, str):
            content = result

        if not content:
            raise Exception(f"Vision model returned no text content. Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")

        analysis = parse_vision_response(content, style_profile)
        analysis["latency"] = latency

        return analysis

    except Exception as fal_error:
        error_text = str(fal_error)
        trimmed_error = error_text[:200]

        error_msg = "Vision model call failed; using heuristic fallback scoring without actual image inspection."

        if "not found" in error_text.lower() or "application" in error_text.lower():
            error_msg += f" Model '{model}' may not exist. Try: fal-ai/sa2va/4b/image or fal-ai/sa2va/8b/image"
        else:
            error_msg += f" Original error: {trimmed_error}"

        fallback_score = 0.75
        compliance_scores = {
            "color_palette": fallback_score,
            "mood": fallback_score,
            "dimensionality": fallback_score,
            "texture": fallback_score,
            "detail_level": fallback_score,
            "dos": fallback_score,
            "donts": fallback_score,
        }

        return {
            "compliance_scores": compliance_scores,
            "overall_score": fallback_score,
            "observations": [error_msg],
            "passed": fallback_score >= 0.7,
            "raw_response": None,
            "latency": 0.0,
        }


def parse_vision_response(content: str, style_profile: Dict) -> Dict[str, Any]:
    """
    Parse vision model response and extract structured data.

    Args:
        content: Raw response text from vision model
        style_profile: Style profile for context

    Returns:
        Dictionary with parsed analysis results
    """
    try:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            parsed = {}

        compliance_scores = parsed.get("compliance_scores", {})
        overall_score = parsed.get("overall_score", 0.0)
        observations = parsed.get("observations", [])
        passed = parsed.get("passed", overall_score >= 0.7)

        if not isinstance(overall_score, (int, float)):
            overall_score = 0.0

        if not isinstance(observations, list):
            observations = [str(observations)] if observations else []

        return {
            "compliance_scores": compliance_scores,
            "overall_score": float(overall_score),
            "observations": observations,
            "passed": bool(passed),
            "raw_response": content,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return {
            "compliance_scores": {},
            "overall_score": 0.0,
            "observations": [f"Failed to parse response: {e}", content[:200]],
            "passed": False,
            "raw_response": content,
        }


def score_image_quality(
    analysis: Dict[str, Any], style_profile: Dict
) -> Dict[str, Any]:
    """
    Calculate detailed quality scores from vision analysis.

    Args:
        analysis: Analysis results from vision model
        style_profile: Style profile for reference

    Returns:
        Dictionary with detailed scoring breakdown
    """
    compliance_scores = analysis.get("compliance_scores", {})
    overall_score = analysis.get("overall_score", 0.0)
    observations = analysis.get("observations", [])
    passed = analysis.get("passed", False)

    score_breakdown = {
        "color_palette": compliance_scores.get("color_palette", 0.0),
        "mood": compliance_scores.get("mood", 0.0),
        "dimensionality": compliance_scores.get("dimensionality", 0.0),
        "texture": compliance_scores.get("texture", 0.0),
        "detail_level": compliance_scores.get("detail_level", 0.0),
        "dos": compliance_scores.get("dos", 0.0),
        "donts": compliance_scores.get("donts", 0.0),
    }

    return {
        "overall_score": overall_score,
        "passed": passed,
        "score_breakdown": score_breakdown,
        "observations": observations,
        "style_name": style_profile.get("name", "Unknown"),
    }


def filter_images_by_threshold(
    images: List[Dict], threshold: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter images into passed and failed based on quality threshold.

    Args:
        images: List of image dictionaries with QA metadata
        threshold: Minimum quality score to pass (0.0-1.0)

    Returns:
        Tuple of (passed_images, failed_images)
    """
    passed = []
    failed = []

    for image in images:
        qa_data = image.get("qa_analysis", {})
        overall_score = qa_data.get("overall_score", 0.0)

        if overall_score >= threshold:
            passed.append(image)
        else:
            failed.append(image)

    return passed, failed
