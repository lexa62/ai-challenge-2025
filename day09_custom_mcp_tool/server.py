import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Service")

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode_location(location_name: str) -> dict:
    """
    Geocode a location name to get coordinates.

    Args:
        location_name: City name or location to search for

    Returns:
        Dictionary with latitude, longitude, and location name, or None if not found
    """
    try:
        params = {"name": location_name, "count": 1, "format": "json"}
        response = requests.get(GEOCODING_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            return {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result["name"],
                "country": result.get("country", ""),
                "timezone": result.get("timezone", "auto"),
            }
        return None
    except requests.RequestException as e:
        raise Exception(f"Failed to geocode location: {e}")


def _get_weather_forecast(latitude: float, longitude: float, timezone: str, units: str = "celsius") -> dict:
    """
    Get weather forecast for given coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        timezone: Timezone identifier
        units: Temperature units - "celsius" or "fahrenheit"

    Returns:
        Dictionary with weather forecast data
    """
    try:
        temp_unit = "fahrenheit" if units.lower() == "fahrenheit" else "celsius"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": timezone,
            "temperature_unit": temp_unit,
            "precipitation_unit": "mm",
        }

        response = requests.get(FORECAST_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "daily" in data:
            daily = data["daily"]
            return {
                "max_temp": daily["temperature_2m_max"][0],
                "min_temp": daily["temperature_2m_min"][0],
                "precipitation": daily["precipitation_sum"][0],
                "weather_code": daily["weather_code"][0],
                "date": daily["time"][0],
                "units": units,
            }
        return None
    except requests.RequestException as e:
        raise Exception(f"Failed to get weather forecast: {e}")


@mcp.tool()
def get_weather(location: str, units: str = "celsius") -> str:
    """
    Get current weather information for a specified location using real weather data.

    Args:
        location: City name or location (e.g., "New York", "London", "Tokyo", "Warsaw")
        units: Temperature units - "celsius" or "fahrenheit" (default: "celsius")

    Returns:
        Weather information including temperature, condition, and precipitation
    """
    try:
        geocode_result = _geocode_location(location)
        if not geocode_result:
            return f"Location '{location}' not found. Please try a different city name."

        location_name = geocode_result["name"]
        country = geocode_result.get("country", "")
        if country:
            location_display = f"{location_name}, {country}"
        else:
            location_display = location_name

        forecast = _get_weather_forecast(
            geocode_result["latitude"],
            geocode_result["longitude"],
            geocode_result["timezone"],
            units
        )

        if not forecast:
            return f"Failed to retrieve weather data for {location_display}."

        temp_unit = "°F" if units.lower() == "fahrenheit" else "°C"
        condition = WMO_WEATHER_CODES.get(forecast["weather_code"], "Unknown")

        result = (
            f"Weather forecast for {location_display} ({forecast['date']}):\n"
            f"  High: {forecast['max_temp']:.1f}{temp_unit}\n"
            f"  Low: {forecast['min_temp']:.1f}{temp_unit}\n"
            f"  Condition: {condition}\n"
            f"  Precipitation: {forecast['precipitation']:.1f} mm"
        )

        return result

    except Exception as e:
        return f"Error retrieving weather data: {str(e)}"


if __name__ == "__main__":
    mcp.run()

