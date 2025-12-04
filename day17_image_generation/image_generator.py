import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import fal_client
import requests

from utils.config import settings


@dataclass
class GenerationResult:
    model_name: str
    input_parameters: Dict[str, Any]
    response_latency: float
    cost_estimate: Optional[float]
    image_url: Optional[str]
    request_id: Optional[str] = None


def estimate_cost(
    model_id: str,
    api_key: str,
    num_images: int = 1,
    image_size: Optional[str] = None,
) -> Optional[float]:
    """
    Estimate the cost for an image generation request using fal.ai Platform API.

    Args:
        model_id: The model identifier (e.g., "fal-ai/flux/dev")
        api_key: The fal.ai API key
        num_images: Number of images to generate
        image_size: Image size (for calculating billing units)

    Returns:
        Estimated cost in USD, or None if unavailable
    """
    try:
        headers = {"Authorization": f"Key {api_key}"}

        # Calculate expected billing units
        # Most models bill per image, with resolution factors
        unit_quantity = num_images

        # Try to get pricing estimate using unit_price method
        estimate_url = "https://api.fal.ai/v1/models/pricing/estimate"
        payload = {
            "method": "unit_price",
            "endpoints": [model_id],
            "unit_quantities": [unit_quantity],
        }

        response = requests.post(estimate_url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            # Check for estimates array
            if "estimates" in data and len(data["estimates"]) > 0:
                estimate = data["estimates"][0]
                if "total_cost" in estimate:
                    return float(estimate["total_cost"])
                elif "unit_price" in estimate:
                    unit_price = float(estimate["unit_price"])
                    unit_qty = float(estimate.get("unit_quantity", unit_quantity))
                    return unit_price * unit_qty
            # Check for direct cost fields
            elif "total_cost" in data:
                return float(data["total_cost"])
            elif "cost" in data:
                return float(data["cost"])

        # If estimate endpoint fails, try getting pricing directly
        pricing_url = f"https://api.fal.ai/v1/models/pricing"
        params = {"endpoint_ids": model_id}
        pricing_response = requests.get(pricing_url, params=params, headers=headers, timeout=10)

        if pricing_response.status_code == 200:
            pricing_data = pricing_response.json()
            # Handle different response formats
            if isinstance(pricing_data, list) and len(pricing_data) > 0:
                price_info = pricing_data[0]
                if "unit_price" in price_info:
                    unit_price = float(price_info["unit_price"])
                    return unit_price * unit_quantity
            elif isinstance(pricing_data, dict):
                if "unit_price" in pricing_data:
                    unit_price = float(pricing_data["unit_price"])
                    return unit_price * unit_quantity
                elif "pricing" in pricing_data:
                    price_info = pricing_data["pricing"]
                    if isinstance(price_info, list) and len(price_info) > 0:
                        if "unit_price" in price_info[0]:
                            unit_price = float(price_info[0]["unit_price"])
                            return unit_price * unit_quantity
    except requests.RequestException:
        # Network or API errors - silently fail
        pass
    except (ValueError, KeyError, TypeError):
        # Data parsing errors - silently fail
        pass
    except Exception:
        # Any other errors - silently fail
        pass

    return None


def generate_image(
    model_id: str,
    prompt: str,
    size: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> GenerationResult:
    """
    Generate an image using fal.ai via fal_client library.

    Args:
        model_id: The model identifier (e.g., "fal-ai/flux/dev")
        prompt: Text description of the image to generate
        size: Image size (e.g., "square_hd", "landscape_4_3", or "1024x1024")
        num_inference_steps: Number of inference steps (quality parameter)
        seed: Random seed for reproducibility
        api_key: fal.ai API key (defaults to settings.fal_api_key)
        timeout: Maximum time to wait for completion in seconds (not used with fal_client)
        poll_interval: Time between status checks in seconds (not used with fal_client)

    Returns:
        GenerationResult with all logged information

    Raises:
        ValueError: If API key is missing or invalid
        Exception: If API request fails
    """
    if api_key is None:
        api_key = settings.fal_api_key

    if not api_key:
        raise ValueError("FAL_API_KEY is required. Set it in your .env file.")

    start_time = time.time()

    # Prepare input parameters
    input_params: Dict[str, Any] = {"prompt": prompt}
    if size:
        input_params["image_size"] = size
    if num_inference_steps is not None:
        input_params["num_inference_steps"] = num_inference_steps
    if seed is not None:
        input_params["seed"] = seed

    try:
        # Set API key for fal_client (it reads from FAL_KEY env var)
        # Temporarily set it if not already set
        original_key = os.environ.get("FAL_KEY")
        os.environ["FAL_KEY"] = api_key

        # Use fal_client.subscribe which handles queuing and polling automatically
        result = fal_client.subscribe(
            model_id,
            arguments=input_params,
        )

        # Restore original key if it existed
        if original_key is not None:
            os.environ["FAL_KEY"] = original_key
        elif "FAL_KEY" in os.environ:
            del os.environ["FAL_KEY"]

        latency = time.time() - start_time

        # Extract image URL from result
        image_url = None
        request_id = None

        # Handle different result formats
        if isinstance(result, dict):
            # Extract request_id if available
            request_id = result.get("request_id")

            # Extract image URL
            if "images" in result and len(result["images"]) > 0:
                image = result["images"][0]
                if isinstance(image, dict):
                    image_url = image.get("url")
                elif isinstance(image, str):
                    image_url = image
            elif "image" in result:
                image_obj = result["image"]
                if isinstance(image_obj, dict):
                    image_url = image_obj.get("url")
                elif isinstance(image_obj, str):
                    image_url = image_obj
            elif "data" in result:
                # Some responses wrap data in a "data" field
                data = result["data"]
                if "images" in data and len(data["images"]) > 0:
                    image = data["images"][0]
                    if isinstance(image, dict):
                        image_url = image.get("url")
                    elif isinstance(image, str):
                        image_url = image
        elif hasattr(result, "images"):
            # Result might be an object with images attribute
            if len(result.images) > 0:
                image = result.images[0]
                if hasattr(image, "url"):
                    image_url = image.url
                elif isinstance(image, dict):
                    image_url = image.get("url")
                elif isinstance(image, str):
                    image_url = image
            if hasattr(result, "request_id"):
                request_id = result.request_id

        # Estimate cost
        cost_estimate = estimate_cost(
            model_id=model_id,
            api_key=api_key,
            num_images=1,
            image_size=size,
        )

        return GenerationResult(
            model_name=model_id,
            input_parameters=input_params,
            response_latency=latency,
            cost_estimate=cost_estimate,
            image_url=image_url,
            request_id=request_id,
        )

    except Exception as e:
        raise Exception(f"Failed to generate image: {e}") from e
