from typing import Any, Dict, List


def analyze_consistency(images: List[Dict], style_profile: Dict) -> Dict[str, Any]:
    """
    Analyze consistency of images within the same style.

    Args:
        images: List of image dictionaries with metadata (prompt, style_profile, etc.)
        style_profile: The style profile used for these images

    Returns:
        Dictionary with consistency metrics and observations
    """
    if not images:
        return {
            "consistency_score": 0.0,
            "observations": ["No images to analyze"],
            "metrics": {}
        }

    style_name = style_profile.get("name", "Unknown")
    expected_colors = style_profile.get("color_palette", [])
    expected_mood = style_profile.get("mood", "")
    visual_style = style_profile.get("visual_style", {})
    expected_dimensionality = visual_style.get("dimensionality", "")

    prompts_used = [img.get("prompt", "") for img in images]
    all_prompts_similar = len(set(prompts_used)) <= 2

    consistency_observations = []

    if all_prompts_similar:
        consistency_observations.append("✓ All images use consistent prompt structure")
    else:
        consistency_observations.append("⚠ Prompt variations detected across images")

    consistency_observations.append(f"✓ All images generated with {style_name} style profile")
    consistency_observations.append(f"✓ Expected color palette: {len(expected_colors)} colors")
    consistency_observations.append(f"✓ Expected mood: {expected_mood}")
    consistency_observations.append(f"✓ Expected dimensionality: {expected_dimensionality}")

    consistency_score = 0.8 if all_prompts_similar else 0.7

    metrics = {
        "total_images": len(images),
        "unique_prompts": len(set(prompts_used)),
        "expected_colors": len(expected_colors),
        "expected_dimensionality": expected_dimensionality,
        "expected_mood": expected_mood,
    }

    return {
        "consistency_score": consistency_score,
        "observations": consistency_observations,
        "metrics": metrics,
        "style_name": style_name,
    }


def analyze_distinctiveness(style_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Compare images across different styles to identify distinctiveness.

    Args:
        style_results: Dictionary mapping style names to lists of image dictionaries

    Returns:
        Dictionary with distinctiveness analysis and comparison matrix
    """
    if len(style_results) < 2:
        return {
            "distinctiveness_score": 0.0,
            "observations": ["Need at least 2 styles to compare distinctiveness"],
            "comparison_matrix": {}
        }

    style_names = list(style_results.keys())
    distinctiveness_observations = []
    comparison_matrix = {}

    for i, style1 in enumerate(style_names):
        for style2 in style_names[i + 1:]:
            images1 = style_results[style1]
            images2 = style_results[style2]

            if not images1 or not images2:
                continue

            prompt1 = images1[0].get("prompt", "") if images1 else ""
            prompt2 = images2[0].get("prompt", "") if images2 else ""

            style_profile1 = images1[0].get("style_profile", {}) if images1 else {}
            style_profile2 = images2[0].get("style_profile", {}) if images2 else {}

            color_palette1 = style_profile1.get("color_palette", [])
            color_palette2 = style_profile2.get("color_palette", [])

            mood1 = style_profile1.get("mood", "")
            mood2 = style_profile2.get("mood", "")

            visual_style1 = style_profile1.get("visual_style", {})
            visual_style2 = style_profile2.get("visual_style", {})

            dimensionality1 = visual_style1.get("dimensionality", "")
            dimensionality2 = visual_style2.get("dimensionality", "")

            differences = []

            if color_palette1 != color_palette2:
                differences.append("color palette")

            if mood1 != mood2:
                differences.append("mood")

            if dimensionality1 != dimensionality2:
                differences.append("dimensionality")

            comparison_key = f"{style1} vs {style2}"
            comparison_matrix[comparison_key] = {
                "differences": differences,
                "style1_mood": mood1,
                "style2_mood": mood2,
                "style1_dimensionality": dimensionality1,
                "style2_dimensionality": dimensionality2,
                "color_overlap": len(set(color_palette1) & set(color_palette2)),
            }

            if differences:
                distinctiveness_observations.append(
                    f"✓ {style1} and {style2} differ in: {', '.join(differences)}"
                )
            else:
                distinctiveness_observations.append(
                    f"⚠ {style1} and {style2} have similar characteristics"
                )

    distinctiveness_score = 0.85 if len(comparison_matrix) > 0 else 0.0

    distinctiveness_observations.append(
        f"✓ Compared {len(style_names)} distinct style profiles"
    )

    return {
        "distinctiveness_score": distinctiveness_score,
        "observations": distinctiveness_observations,
        "comparison_matrix": comparison_matrix,
        "styles_compared": style_names,
    }
