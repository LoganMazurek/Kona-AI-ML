"""
Weather data module for adding historical temperature and rain data to event datasets.

This module provides functionality to fetch historical weather data from Open-Meteo's
free ERA5 archive API, which includes historical data for temperature and precipitation.
"""

import pandas as pd
import requests
import time
from datetime import datetime, timezone
import os
from typing import Dict, Any, Optional, Tuple
import logging

# Dual import pattern for package and script execution
try:
    from Kona_AI_ML.weather_providers import get_weather_provider, WeatherProvider
except ImportError:
    from weather_providers import get_weather_provider, WeatherProvider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from uszipcode import SearchEngine
    USZIPCODE_AVAILABLE = True
except ImportError:
    USZIPCODE_AVAILABLE = False
    logger.warning("uszipcode package not available. Install with: pip install uszipcode")

# Initialize ZIP code search engine (singleton pattern)
_zip_search_engine = None

def get_zip_search_engine():
    """Get or create the ZIP code search engine singleton."""
    global _zip_search_engine
    if _zip_search_engine is None and USZIPCODE_AVAILABLE:
        _zip_search_engine = SearchEngine()
    return _zip_search_engine

class WeatherDataEnricher:
    """Class to enrich event datasets with historical weather data using Open-Meteo."""
    
    def __init__(self, provider: str = 'open-meteo', api_key: Optional[str] = None):
        """
        Initialize the WeatherDataEnricher.
        
        Args:
            provider: Weather provider to use (default: 'open-meteo')
            api_key: Deprecated, kept for backward compatibility
        """
        self.provider_name = provider
        
        try:
            self.weather_provider = get_weather_provider(provider, api_key)
            logger.info(f"Initialized weather provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize weather provider {provider}: {e}")
            self.weather_provider = None
        
    def _get_coordinates_from_zip(self, zip_code: str) -> Tuple[float, float]:
        """
        Get latitude and longitude from zip code using uszipcode database.
        
        Args:
            zip_code: US zip code
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Try uszipcode first
        if USZIPCODE_AVAILABLE:
            search = get_zip_search_engine()
            if search:
                try:
                    zipcode = search.by_zipcode(str(zip_code))
                    if zipcode and zipcode.lat and zipcode.lng:
                        logger.debug(f"Found coordinates for ZIP {zip_code}: {zipcode.lat}, {zipcode.lng}")
                        return zipcode.lat, zipcode.lng
                except Exception as e:
                    logger.warning(f"Error looking up ZIP {zip_code}: {e}")
        
        # Fallback coordinates (Tinley Park/Chicago area) if lookup fails
        logger.debug(f"Using fallback coordinates for zip {zip_code}")
        return 41.8781, -87.6298
    
    def get_historical_weather(self, lat: float, lon: float, event_datetime: datetime) -> Optional[Dict[str, Any]]:
        """
        Get historical weather data for a specific location and datetime.
        
        Args:
            lat: Latitude
            lon: Longitude
            event_datetime: Event datetime
            
        Returns:
            Dict with weather data or None if unavailable
        """
        if not self.weather_provider:
            logger.warning("No weather provider available, returning None values")
            return None
            
        return self.weather_provider.get_historical_weather(lat, lon, event_datetime)
    
    def enrich_dataset(self, df: pd.DataFrame, 
                      datetime_column: str = 'Event Start Date Time',
                      zip_column: str = 'Zip Code',
                      lat_column: Optional[str] = None,
                      lon_column: Optional[str] = None) -> pd.DataFrame:
        """
        Enrich a dataset with weather data.
        
        Args:
            df: DataFrame containing event data
            datetime_column: Name of column containing event datetime
            zip_column: Name of column containing zip code (used if lat/lon not provided)
            lat_column: Name of column containing latitude (optional)
            lon_column: Name of column containing longitude (optional)
            
        Returns:
            DataFrame with added 'Temperature_F' and 'Is_Raining' columns
        """
        if not self.weather_provider:
            logger.error("No weather provider available. Cannot enrich dataset with weather data.")
            # Return dataset with empty weather columns
            df = df.copy()
            df['Temperature_F'] = None
            df['Is_Raining'] = None
            df['Precipitation_Inches'] = None
            df['Cloud_Cover_Percent'] = None
            df['Wind_Speed_MPH'] = None
            df['Feels_Like_Temp_F'] = None
            return df
        
        logger.info(f"Enriching dataset with {len(df)} rows with weather data using {self.provider_name}...")
        
        df = df.copy()
        df['Temperature_F'] = None
        df['Is_Raining'] = None
        df['Precipitation_Inches'] = None
        df['Cloud_Cover_Percent'] = None
        df['Wind_Speed_MPH'] = None
        df['Feels_Like_Temp_F'] = None
        
        # Convert datetime column to datetime if it isn't already
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # Process each row
        count = 0
        for row_idx, row in df.iterrows():
            try:
                event_datetime = row[datetime_column]
                
                # Skip if datetime is missing
                if pd.isna(event_datetime):
                    continue
                
                # Get coordinates
                if lat_column and lon_column and not pd.isna(row[lat_column]) and not pd.isna(row[lon_column]):
                    lat, lon = row[lat_column], row[lon_column]
                elif zip_column and not pd.isna(row[zip_column]):
                    lat, lon = self._get_coordinates_from_zip(str(row[zip_column]))
                else:
                    logger.warning(f"No location data available for row {row_idx}")
                    continue
                
                # Get weather data
                weather_data = self.get_historical_weather(lat, lon, event_datetime)
                
                # Update dataframe
                if weather_data:
                    df.at[row_idx, 'Temperature_F'] = weather_data.get('temperature_f')
                    df.at[row_idx, 'Is_Raining'] = weather_data.get('is_raining')
                    df.at[row_idx, 'Precipitation_Inches'] = weather_data.get('precipitation_inches')
                    df.at[row_idx, 'Cloud_Cover_Percent'] = weather_data.get('cloud_cover_percent')
                    df.at[row_idx, 'Wind_Speed_MPH'] = weather_data.get('wind_speed_mph')
                    df.at[row_idx, 'Feels_Like_Temp_F'] = weather_data.get('feels_like_temp_f')
                
                count += 1
                if count % 10 == 0:
                    logger.info(f"Processed {count}/{len(df)} rows")
                    
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {e}")
                continue
        
        logger.info("Weather enrichment complete!")
        return df


def add_weather_to_csv(input_csv_path: str, output_csv_path: str, 
                      provider: str = 'open-meteo', api_key: Optional[str] = None):
    """
    Convenience function to add weather data to a CSV file.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to save enriched CSV file
        provider: Weather provider (default: 'open-meteo')
        api_key: Deprecated, kept for backward compatibility
    """
    # Read the CSV
    df = pd.read_csv(input_csv_path)
    
    # Initialize enricher
    enricher = WeatherDataEnricher(provider=provider, api_key=api_key)
    
    # Enrich with weather data
    enriched_df = enricher.enrich_dataset(df)
    
    # Save to new CSV
    enriched_df.to_csv(output_csv_path, index=False)
    
    logger.info(f"Enriched dataset saved to {output_csv_path}")
    
    # Print summary statistics
    print(f"\nWeather Data Summary:")
    print(f"\nTemperature (°F):")
    print(enriched_df['Temperature_F'].describe())
    print(f"\nFeels Like Temperature (°F):")
    print(enriched_df['Feels_Like_Temp_F'].describe())
    print(f"\nPrecipitation (inches):")
    print(enriched_df['Precipitation_Inches'].describe())
    print(f"\nCloud Cover (%):")
    print(enriched_df['Cloud_Cover_Percent'].describe())
    print(f"\nWind Speed (mph):")
    print(enriched_df['Wind_Speed_MPH'].describe())
    rain_percentage = (enriched_df['Is_Raining'] == True).mean() * 100
    print(f"\nPercentage of events with rain: {rain_percentage:.1f}%")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Add weather data to event CSV file using Open-Meteo')
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    add_weather_to_csv(args.input_csv, args.output_csv)