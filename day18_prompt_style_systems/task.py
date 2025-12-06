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
from day18_prompt_style_systems.image_comparison import analyze_consistency, analyze_distinctiveness
from day18_prompt_style_systems.prompt_template import build_prompt, get_aspect_ratio_size, load_style_profiles
from utils.config import settings
from utils.helpers import header, info, success, warn

console = Console()

MODEL_ID = "fal-ai/flux/dev"
IMAGES_PER_STYLE = 4


def save_generation_log(generation_log: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save generation logs to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "day18_generations.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(generation_log, f, indent=2, ensure_ascii=False)

    console.print(f"[dim]Generation log saved to: {log_file}[/dim]")


def save_results_markdown(
    subject: str,
    style_results: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    """Save results to a markdown file with image URLs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_file = output_dir / "day18_results.md"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Day 18 — Prompt & Style Systems Results\n\n")
        f.write(f"**Subject:** {subject}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for style_name, images in style_results.items():
            f.write(f"## {style_name}\n\n")

            for i, img in enumerate(images, 1):
                f.write(f"### Image {i}\n\n")
                f.write(f"**Prompt:** {img.get('prompt', 'N/A')}\n\n")
                f.write(f"**Image URL:** {img.get('image_url', 'N/A')}\n\n")
                if img.get('image_url'):
                    f.write(f"![{style_name} Image {i}]({img['image_url']})\n\n")
                f.write("---\n\n")

    console.print(f"[dim]Results saved to: {md_file}[/dim]")


def display_style_profiles(profiles: List[Dict]) -> None:
    """Display style profiles in a table."""
    table = Table(show_header=True, header_style="bold magenta", title="Style Profiles")
    table.add_column("Name", style="cyan", width=20)
    table.add_column("Mood", style="white", width=25)
    table.add_column("Dimensionality", style="green", width=15)
    table.add_column("Detail Level", style="yellow", width=15)
    table.add_column("Colors", style="blue", width=30)

    for profile in profiles:
        visual_style = profile.get("visual_style", {})
        colors = ", ".join(profile.get("color_palette", [])[:3])
        if len(profile.get("color_palette", [])) > 3:
            colors += "..."

        table.add_row(
            profile.get("name", "Unknown"),
            profile.get("mood", "N/A"),
            visual_style.get("dimensionality", "N/A"),
            visual_style.get("detail_level", "N/A"),
            colors,
        )

    console.print()
    console.print(table)
    console.print()


def generate_images_for_style(
    subject: str,
    style_profile: Dict,
    aspect_ratio: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """Generate 4 images for a given style profile."""
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

                image_data = {
                    "prompt": prompt,
                    "style_profile": style_profile,
                    "style_name": style_profile.get("name"),
                    "aspect_ratio": aspect_ratio,
                    "size": size,
                    "seed": seed,
                    "image_url": result.image_url,
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


def display_image_grids(style_results: Dict[str, List[Dict]]) -> None:
    """Display image URLs in an organized grid format."""
    console.print()
    header("Generated Images")
    console.print()

    for style_name, images in style_results.items():
        console.print(f"[bold cyan]{style_name}[/bold cyan]")
        console.print()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Image #", style="cyan", width=10)
        table.add_column("Image URL", style="blue", no_wrap=True)
        table.add_column("Seed", style="yellow", width=10)
        table.add_column("Latency", style="green", width=12)

        for i, img in enumerate(images, 1):
            url = img.get("image_url", "N/A")
            seed = img.get("seed", "N/A")
            latency = f"{img.get('latency', 0):.2f}s"

            table.add_row(str(i), url, str(seed), latency)

        console.print(table)
        console.print()

        console.print("[bold]Full Image URLs:[/bold]")
        for i, img in enumerate(images, 1):
            url = img.get("image_url", "N/A")
            if url != "N/A":
                console.print(f"  Image {i}: [link={url}]{url}[/link]")
        console.print()


def display_consistency_analysis(style_results: Dict[str, List[Dict]]) -> None:
    """Display consistency analysis for each style."""
    console.print()
    header("Consistency Analysis (Within Styles)")
    console.print()

    for style_name, images in style_results.items():
        style_profile = images[0].get("style_profile", {}) if images else {}
        analysis = analyze_consistency(images, style_profile)

        panel_content = f"[bold]Style:[/bold] {analysis.get('style_name', 'Unknown')}\n"
        panel_content += f"[bold]Consistency Score:[/bold] {analysis.get('consistency_score', 0.0):.2f}\n\n"
        panel_content += "[bold]Observations:[/bold]\n"

        for obs in analysis.get("observations", []):
            panel_content += f"  {obs}\n"

        metrics = analysis.get("metrics", {})
        if metrics:
            panel_content += "\n[bold]Metrics:[/bold]\n"
            for key, value in metrics.items():
                panel_content += f"  {key}: {value}\n"

        console.print(
            Panel(
                panel_content,
                title=f"[bold green]{style_name} Consistency[/bold green]",
                border_style="green",
            )
        )
        console.print()


def display_distinctiveness_analysis(style_results: Dict[str, List[Dict]]) -> None:
    """Display distinctiveness analysis between styles."""
    console.print()
    header("Distinctiveness Analysis (Between Styles)")
    console.print()

    analysis = analyze_distinctiveness(style_results)

    panel_content = f"[bold]Distinctiveness Score:[/bold] {analysis.get('distinctiveness_score', 0.0):.2f}\n\n"
    panel_content += "[bold]Observations:[/bold]\n"

    for obs in analysis.get("observations", []):
        panel_content += f"  {obs}\n"

    comparison_matrix = analysis.get("comparison_matrix", {})
    if comparison_matrix:
        panel_content += "\n[bold]Comparison Matrix:[/bold]\n"
        for comparison, data in comparison_matrix.items():
            panel_content += f"\n  {comparison}:\n"
            differences = data.get("differences", [])
            if differences:
                panel_content += f"    Differences: {', '.join(differences)}\n"
            panel_content += f"    Color overlap: {data.get('color_overlap', 0)} colors\n"

    console.print(
        Panel(
            panel_content,
            title="[bold blue]Style Distinctiveness[/bold blue]",
            border_style="blue",
        )
    )
    console.print()


def run() -> None:
    """Main entry point for Day 18 task."""
    api_key = settings.fal_api_key
    if not api_key:
        console.print("[red]FAL_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    header("Day 18 — Prompt & Style Systems")
    console.print()
    console.print("[dim]Build a system that turns ad-hoc prompts into controlled, reusable visual styles.[/dim]")
    console.print()

    profile_path = Path(__file__).parent / "style_profiles.json"

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

    display_style_profiles(profiles)

    subject = Prompt.ask("Enter the base subject for image generation", default="a coffee cup")
    aspect_ratio = Prompt.ask(
        "Choose aspect ratio",
        choices=["square", "landscape", "portrait"],
        default="square",
    )

    console.print()
    console.print(f"[bold]Subject:[/bold] {subject}")
    console.print(f"[bold]Aspect Ratio:[/bold] {aspect_ratio}")
    console.print()

    style_results = {}
    generation_log = []

    for profile in profiles:
        style_name = profile.get("name", "Unknown")
        console.print(f"[yellow]Generating images for {style_name}...[/yellow]")

        images = generate_images_for_style(subject, profile, aspect_ratio, api_key)

        if images:
            style_results[style_name] = images
            generation_log.extend(images)
            success(f"Generated {len(images)} images for {style_name}")
        else:
            warn(f"No images generated for {style_name}")

        console.print()

    if not style_results:
        console.print("[red]No images were generated. Exiting.[/red]")
        return

    display_image_grids(style_results)
    display_consistency_analysis(style_results)
    display_distinctiveness_analysis(style_results)

    output_dir = Path("outputs")
    save_generation_log(generation_log, output_dir)
    save_results_markdown(subject, style_results, output_dir)

    console.print()
    success("Image generation and analysis complete!")
    console.print()
