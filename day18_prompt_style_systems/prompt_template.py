import json
from pathlib import Path
from typing import Dict, List


def load_style_profiles(path: str) -> List[Dict]:
    """Load and validate JSON style profiles."""
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Style profiles file not found: {path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    if not isinstance(profiles, list):
        raise ValueError("Style profiles must be a JSON array")

    required_fields = ["name", "color_palette", "mood", "visual_style", "dos", "donts", "style_description"]
    for i, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {i} must be a JSON object")
        for field in required_fields:
            if field not in profile:
                raise ValueError(f"Profile {i} missing required field: {field}")

    return profiles


def get_aspect_ratio_size(aspect_ratio: str) -> str:
    """Map aspect ratios to fal.ai size formats."""
    aspect_ratio_lower = aspect_ratio.lower().strip()

    mapping = {
        "square": "square_hd",
        "landscape": "landscape_4_3",
        "portrait": "portrait_4_3",
        "1:1": "square_hd",
        "4:3": "landscape_4_3",
        "3:4": "portrait_4_3",
    }

    return mapping.get(aspect_ratio_lower, "square_hd")


def build_prompt(subject: str, style_profile: Dict, aspect_ratio: str) -> str:
    """
    Build a prompt by combining base subject + style description + aspect ratio guidance.

    Args:
        subject: The base subject of the image
        style_profile: Dictionary containing style profile data
        aspect_ratio: Aspect ratio string (e.g., "square", "landscape", "portrait")

    Returns:
        Formatted prompt string
    """
    style_desc = style_profile.get("style_description", "")
    mood = style_profile.get("mood", "")
    colors = ", ".join(style_profile.get("color_palette", []))
    visual_style = style_profile.get("visual_style", {})
    dos = style_profile.get("dos", [])
    donts = style_profile.get("donts", [])

    dimensionality = visual_style.get("dimensionality", "")
    texture = visual_style.get("texture", "")
    detail_level = visual_style.get("detail_level", "")

    prompt_parts = [subject]

    if style_desc:
        prompt_parts.append(f"Style: {style_desc}")

    if mood:
        prompt_parts.append(f"Mood: {mood}")

    if colors:
        prompt_parts.append(f"Color palette: {colors}")

    if dimensionality:
        prompt_parts.append(f"{dimensionality} design")

    if texture:
        prompt_parts.append(f"{texture} texture")

    if detail_level:
        prompt_parts.append(f"{detail_level} detail level")

    if dos:
        dos_text = ", ".join(dos[:3])
        prompt_parts.append(f"Requirements: {dos_text}")

    if donts:
        donts_text = ", ".join(donts[:2])
        prompt_parts.append(f"Avoid: {donts_text}")

    aspect_ratio_note = ""
    if aspect_ratio.lower() == "landscape":
        aspect_ratio_note = "wide horizontal composition"
    elif aspect_ratio.lower() == "portrait":
        aspect_ratio_note = "tall vertical composition"
    else:
        aspect_ratio_note = "square composition"

    if aspect_ratio_note:
        prompt_parts.append(f"Format: {aspect_ratio_note}")

    return ", ".join(prompt_parts)
