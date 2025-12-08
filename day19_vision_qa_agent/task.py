import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.table import Table

from day17_image_generation.image_generator import generate_image
from day18_prompt_style_systems.prompt_template import (
    build_prompt,
    get_aspect_ratio_size,
    load_style_profiles,
)
from day19_vision_qa_agent.vision_qa import (
    analyze_image_with_vision,
    filter_images_by_threshold,
    score_image_quality,
)
from utils.config import settings
from utils.helpers import header, info, success, warn

console = Console()

MODEL_ID = "fal-ai/flux/dev"
IMAGES_PER_STYLE = 4


def evaluate_image_quality(
    image_data: Dict[str, Any],
    style_profile: Dict,
    api_key: str = None,
    vision_model: str = None,
    fal_api_key: str = None,
) -> Dict[str, Any]:
    """
    Evaluate image quality using fal.ai vision model.

    Args:
        image_data: Image metadata dictionary
        style_profile: Style profile to evaluate against
        api_key: Not used (kept for compatibility)
        vision_model: Vision model to use (optional, defaults to fal-ai/sa2va/8b/image)
        fal_api_key: fal.ai API key

    Returns:
        Image data dictionary with added QA analysis
    """
    image_url = image_data.get("image_url")
    image_bytes = image_data.get("image_bytes")

    if not image_url and not image_bytes:
        warn(f"Image {image_data.get('seed', 'unknown')} has no URL or image data, skipping QA")
        image_data["qa_analysis"] = {
            "overall_score": 0.0,
            "passed": False,
            "error": "No image URL or image data",
        }
        return image_data

    try:
        analysis = analyze_image_with_vision(
            image_url=image_url,
            image_bytes=image_bytes,
            style_profile=style_profile,
            api_key=api_key,
            model=vision_model,
            fal_api_key=fal_api_key,
        )
        qa_results = score_image_quality(analysis, style_profile)

        image_data["qa_analysis"] = qa_results
        image_data["qa_timestamp"] = datetime.now().isoformat()

    except Exception as e:
        warn(f"QA evaluation failed for image {image_data.get('seed', 'unknown')}: {e}")
        image_data["qa_analysis"] = {
            "overall_score": 0.0,
            "passed": False,
            "error": str(e),
        }

    return image_data


def run_qa_pipeline(
    images: List[Dict[str, Any]],
    style_profile: Dict,
    api_key: str = None,
    vision_model: str = None,
    fal_api_key: str = None,
) -> List[Dict[str, Any]]:
    """
    Run QA pipeline on all images using fal.ai vision model.

    Args:
        images: List of image data dictionaries
        style_profile: Style profile to evaluate against
        api_key: Not used (kept for compatibility)
        vision_model: Vision model to use (optional)
        fal_api_key: fal.ai API key

    Returns:
        List of image data dictionaries with QA analysis
    """
    evaluated_images = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Evaluating images with {vision_model}...",
            total=len(images),
        )

        for i, image_data in enumerate(images, 1):
            seed = image_data.get("seed", "unknown")
            progress.update(task, description=f"Evaluating image {i}/{len(images)} (seed: {seed})...")

            evaluated = evaluate_image_quality(
                image_data, style_profile, api_key, vision_model, fal_api_key
            )
            evaluated_images.append(evaluated)

            qa = evaluated.get("qa_analysis", {})
            score = qa.get("overall_score", 0.0)
            progress.update(task, advance=1, description=f"Evaluated {i}/{len(images)} (score: {score:.2f})")

    return evaluated_images


def generate_images_for_style(
    subject: str,
    style_profile: Dict,
    aspect_ratio: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Generate images for a given style profile."""
    images = []
    size = get_aspect_ratio_size(aspect_ratio)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating images for {style_profile.get('name', 'Unknown')}...",
            total=IMAGES_PER_STYLE,
        )

        for i in range(IMAGES_PER_STYLE):
            prompt = build_prompt(subject, style_profile, aspect_ratio)
            seed = random.randint(1, 1000000)

            try:
                result = generate_image(
                    model_id=MODEL_ID,
                    prompt=prompt,
                    size=size,
                    seed=seed,
                    api_key=api_key,
                )

                image_bytes = None
                if result.image_url:
                    try:
                        from day19_vision_qa_agent.vision_qa import download_image
                        image_bytes = download_image(result.image_url, api_key)
                    except Exception as download_err:
                        warn(f"Failed to download image immediately: {download_err}")

                image_data = {
                    "prompt": prompt,
                    "style_profile": style_profile,
                    "style_name": style_profile.get("name"),
                    "aspect_ratio": aspect_ratio,
                    "size": size,
                    "seed": seed,
                    "image_url": result.image_url,
                    "image_bytes": image_bytes,
                    "latency": result.response_latency,
                    "cost_estimate": result.cost_estimate,
                    "model_name": result.model_name,
                    "timestamp": datetime.now().isoformat(),
                }

                images.append(image_data)
                progress.update(task, advance=1)

            except Exception as e:
                warn(f"Failed to generate image {i+1}: {e}")
                progress.update(task, advance=1)

    return images


def display_qa_results(
    qa_results: Dict[str, Any],
    threshold: float,
) -> None:
    """
    Display QA results in a formatted table.

    Args:
        qa_results: Dictionary with style_name, passed_images, failed_images, stats
        threshold: Quality threshold used
    """
    console.print()
    header("Vision QA Results")
    console.print()

    style_name = qa_results.get("style_name", "Unknown")
    passed_images = qa_results.get("passed_images", [])
    failed_images = qa_results.get("failed_images", [])
    stats = qa_results.get("stats", {})

    stats_panel = f"[bold]Quality Threshold:[/bold] {threshold:.2f}\n"
    stats_panel += f"[bold]Total Images:[/bold] {stats.get('total', 0)}\n"
    stats_panel += f"[bold green]Passed:[/bold green] {stats.get('passed_count', 0)}\n"
    stats_panel += f"[bold red]Failed:[/bold red] {stats.get('failed_count', 0)}\n"
    stats_panel += f"[bold]Pass Rate:[/bold] {stats.get('pass_rate', 0.0):.1f}%"

    console.print(
        Panel(
            stats_panel,
            title=f"[bold cyan]{style_name} QA Summary[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    if passed_images or failed_images:
        table = Table(show_header=True, header_style="bold magenta", title="Image Quality Scores")
        table.add_column("Image #", style="cyan", width=10)
        table.add_column("Seed", style="yellow", width=10)
        table.add_column("Overall Score", style="green", width=15)
        table.add_column("Status", style="white", width=12)
        table.add_column("Image URL", style="blue", no_wrap=True)

        all_images = passed_images + failed_images
        for i, img in enumerate(all_images, 1):
            qa = img.get("qa_analysis", {})
            score = qa.get("overall_score", 0.0)
            seed = img.get("seed", "N/A")
            url = img.get("image_url", "N/A")
            status = "[green]PASS[/green]" if img in passed_images else "[red]FAIL[/red]"

            table.add_row(str(i), str(seed), f"{score:.2f}", status, url)

        console.print(table)
        console.print()

        console.print("[bold]Full Image URLs:[/bold]")
        for i, img in enumerate(all_images, 1):
            url = img.get("image_url", "N/A")
            if url != "N/A":
                console.print(f"  Image {i}: [link={url}]{url}[/link]")
        console.print()

    if passed_images:
        console.print("[bold green]Passed Images:[/bold green]")
        for i, img in enumerate(passed_images, 1):
            qa = img.get("qa_analysis", {})
            score = qa.get("overall_score", 0.0)
            breakdown = qa.get("score_breakdown", {})
            observations = qa.get("observations", [])
            url = img.get("image_url", "N/A")

            panel_content = f"[bold]Score:[/bold] {score:.2f}\n"
            panel_content += f"[bold]Image URL:[/bold] [link={url}]{url}[/link]\n\n"
            panel_content += "[bold]Score Breakdown:[/bold]\n"
            for criterion, criterion_score in breakdown.items():
                panel_content += f"  {criterion}: {criterion_score:.2f}\n"

            if observations:
                panel_content += "\n[bold]Observations:[/bold]\n"
                for obs in observations[:3]:
                    panel_content += f"  • {obs}\n"

            console.print(
                Panel(
                    panel_content,
                    title=f"[green]Image {i} (Seed: {img.get('seed', 'N/A')})[/green]",
                    border_style="green",
                )
            )
            console.print()

    if failed_images:
        console.print("[bold red]Failed Images:[/bold red]")
        for i, img in enumerate(failed_images, 1):
            qa = img.get("qa_analysis", {})
            score = qa.get("overall_score", 0.0)
            error = qa.get("error")
            observations = qa.get("observations", [])
            url = img.get("image_url", "N/A")

            panel_content = f"[bold]Score:[/bold] {score:.2f}\n"
            panel_content += f"[bold]Image URL:[/bold] [link={url}]{url}[/link]\n"
            if error:
                panel_content += f"[bold red]Error:[/bold red] {error}\n"

            if observations:
                panel_content += "\n[bold]Observations:[/bold]\n"
                for obs in observations[:3]:
                    panel_content += f"  • {obs}\n"

            console.print(
                Panel(
                    panel_content,
                    title=f"[red]Image {i} (Seed: {img.get('seed', 'N/A')})[/red]",
                    border_style="red",
                )
            )
            console.print()


def save_qa_results(
    qa_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save QA results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "day19_qa_results.json"

    def clean_for_json(obj):
        """Remove non-serializable objects (like bytes) from data structure."""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != "image_bytes"}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, bytes):
            return None
        else:
            return obj

    cleaned_results = clean_for_json(qa_results)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

    console.print(f"[dim]QA results saved to: {log_file}[/dim]")


def run() -> None:
    """Main entry point for Day 19 task."""
    fal_api_key = settings.fal_api_key

    if not fal_api_key:
        console.print("[red]FAL_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    header("Day 19 — Vision QA Agent")
    console.print()
    console.print("[dim]Automated vision-based quality assurance for generated images.[/dim]")
    console.print()

    profile_path = Path(__file__).parent.parent / "day18_prompt_style_systems" / "style_profiles.json"

    try:
        profiles = load_style_profiles(str(profile_path))
    except Exception as e:
        console.print(f"[red]Failed to load style profiles: {e}[/red]")
        return

    if not profiles:
        console.print("[red]No style profiles found.[/red]")
        return

    console.print(f"[green]Loaded {len(profiles)} style profile(s)[/green]")
    console.print()

    subject = Prompt.ask("Enter the base subject for image generation", default="a coffee cup")
    aspect_ratio = Prompt.ask(
        "Choose aspect ratio",
        choices=["square", "landscape", "portrait"],
        default="square",
    )

    threshold_input = Prompt.ask(
        "Enter quality threshold (0.0-1.0)",
        default="0.7",
    )
    try:
        threshold = float(threshold_input)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    except ValueError as e:
        console.print(f"[red]Invalid threshold: {e}. Using default 0.7[/red]")
        threshold = 0.7

    vision_model = Prompt.ask(
        "fal.ai vision model (options: fal-ai/sa2va/4b/image, fal-ai/sa2va/8b/image)",
        default="fal-ai/sa2va/4b/image",
    )

    console.print()
    console.print(f"[bold]Subject:[/bold] {subject}")
    console.print(f"[bold]Aspect Ratio:[/bold] {aspect_ratio}")
    console.print(f"[bold]Quality Threshold:[/bold] {threshold:.2f}")
    console.print(f"[bold]Vision Model:[/bold] {vision_model}")
    console.print(f"[bold]Using fal.ai for vision analysis[/bold]")
    console.print()

    profile_choice = Prompt.ask(
        f"Select style profile (1-{len(profiles)})",
        default="1",
    )
    try:
        profile_idx = int(profile_choice) - 1
        if not 0 <= profile_idx < len(profiles):
            raise ValueError("Invalid profile index")
        selected_profile = profiles[profile_idx]
    except (ValueError, IndexError):
        console.print("[yellow]Invalid selection, using first profile[/yellow]")
        selected_profile = profiles[0]

    console.print()
    console.print(f"[bold]Selected Style:[/bold] {selected_profile.get('name', 'Unknown')}")
    console.print()

    console.print("[yellow]Step 1: Generating images...[/yellow]")
    images = generate_images_for_style(subject, selected_profile, aspect_ratio, fal_api_key)

    if not images:
        console.print("[red]No images were generated. Exiting.[/red]")
        return

    success(f"Generated {len(images)} images")
    console.print()

    console.print("[yellow]Step 2: Running vision QA analysis with fal.ai...[/yellow]")
    evaluated_images = run_qa_pipeline(images, selected_profile, None, vision_model, fal_api_key)

    console.print()
    console.print("[yellow]Step 3: Filtering by quality threshold...[/yellow]")
    passed_images, failed_images = filter_images_by_threshold(evaluated_images, threshold)

    stats = {
        "total": len(evaluated_images),
        "passed_count": len(passed_images),
        "failed_count": len(failed_images),
        "pass_rate": (len(passed_images) / len(evaluated_images) * 100) if evaluated_images else 0.0,
    }

    qa_results = {
        "style_name": selected_profile.get("name", "Unknown"),
        "style_profile": selected_profile,
        "subject": subject,
        "aspect_ratio": aspect_ratio,
        "threshold": threshold,
        "vision_model": vision_model,
        "vision_provider": "fal.ai",
        "passed_images": passed_images,
        "failed_images": failed_images,
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }

    display_qa_results(qa_results, threshold)

    output_dir = Path("outputs")
    save_qa_results(qa_results, output_dir)

    console.print()
    success("Vision QA pipeline complete!")
    console.print()
