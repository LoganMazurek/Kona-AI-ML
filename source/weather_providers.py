"""
Weather providers module with Open-Meteo API integration.
Provides weather data for historical events using free Open-Meteo API.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import logging
import requests
import time

logger = logging.getLogger(__name__)


class WeatherProvider:
    """Base class for weather data providers."""
    
    def get_historical_weather(self, lat: float, lon: float, 
                              event_datetime: datetime) -> Optional[Dict[str, Any]]:
        """
        Get historical weather data for a specific location and datetime.
        
        Args:
            lat: Latitude
            lon: Longitude
            event_datetime: Event datetime
            
        Returns:
            Dict with weather data (temperature_f, precipitation_inches, is_raining, 
            cloud_cover_percent, wind_speed_mph, feels_like_temp_f) or None if unavailable
        """
        raise NotImplementedError


class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo weather provider with historical weather API."""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize Open-Meteo provider.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of API retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_count = 0
        self.api_calls_made = 0
    
    def get_historical_weather(self, lat: float, lon: float, 
                              event_datetime: datetime) -> Optional[Dict[str, Any]]:
        """
        Get historical weather data from Open-Meteo API.
        
        Args:
            lat: Latitude
            lon: Longitude
            event_datetime: Event datetime (should be in past)
            
        Returns:
            Dict with weather data or None if unavailable
        """
        try:
            # Format date for API
            event_date = event_datetime.date().isoformat()
            
            # Build API request
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': event_date,
                'end_date': event_date,
                'hourly': 'temperature_2m,precipitation,cloudcover,windspeed_10m,apparent_temperature',
                'temperature_unit': 'fahrenheit',
                'windspeed_unit': 'mph',
                'timezone': 'auto'
            }
            
            # Make API call with retries
            for attempt in range(self.max_retries):
                try:
                    self.request_count += 1
                    response = requests.get(
                        self.BASE_URL,
                        params=params,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        self.api_calls_made += 1
                        data = response.json()
                        
                        # Extract hourly data
                        if 'hourly' in data and data['hourly']:
                            hourly = data['hourly']
                            temps = hourly.get('temperature_2m', [])
                            precips = hourly.get('precipitation', [])
                            clouds = hourly.get('cloudcover', [])
                            winds = hourly.get('windspeed_10m', [])
                            feels_like = hourly.get('apparent_temperature', [])
                            
                            # Get weather for the event's hour (or average if no specific hour)
                            if temps:
                                # Use average values for the day
                                weather_data = {
                                    'temperature_f': sum(temps) / len(temps),
                                    'precipitation_inches': sum(precips) / len(precips) if precips else 0.0,
                                    'is_raining': any(p > 0 for p in precips) if precips else False,
                                    'cloud_cover_percent': sum(clouds) / len(clouds) if clouds else None,
                                    'wind_speed_mph': sum(winds) / len(winds) if winds else None,
                                    'feels_like_temp_f': sum(feels_like) / len(feels_like) if feels_like else None
                                }
                                
                                logger.debug(f"API data for {lat},{lon} on {event_date}: {weather_data['temperature_f']:.1f}F, raining={weather_data['is_raining']}, wind={weather_data['wind_speed_mph']:.1f}mph, cloud={weather_data['cloud_cover_percent']:.0f}%")
                                return weather_data
                        
                        # If no hourly data, return None to fall back to seasonal
                        return None
                    
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    
                    else:
                        logger.warning(f"API returned status {response.status_code}")
                        return None, None
                
                except requests.Timeout:
                    if attempt < self.max_retries - 1:
                        logger.debug(f"Timeout on attempt {attempt+1}, retrying...")
                        time.sleep(1)
                        continue
                    else:
                        logger.warning(f"Timeout after {self.max_retries} attempts")
                        return None
                
                except Exception as e:
                    logger.debug(f"API error on attempt {attempt+1}: {e}")
                    if attempt == self.max_retries - 1:
                        return None
                    time.sleep(1)
            
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error in get_historical_weather: {e}")
            return None


def get_weather_provider(provider_name: str = 'open-meteo', 
                        api_key: Optional[str] = None) -> Optional[WeatherProvider]:
    """
    Factory function to get weather provider instance.
    
    Args:
        provider_name: Name of provider ('open-meteo')
        api_key: Optional API key (for future use)
        
    Returns:
        WeatherProvider instance or None if provider not available
    """
    if provider_name == 'open-meteo':
        return OpenMeteoProvider()
    
    logger.warning(f"Unknown weather provider: {provider_name}")
    return None
