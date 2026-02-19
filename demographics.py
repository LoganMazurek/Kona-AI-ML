"""
Module for retrieving and processing demographic data based on ZIP codes.
Uses Census API for demographic information and uszipcode for geographic metadata.
"""
import pandas as pd
import numpy as np
import requests
from typing import Dict, Optional, Any
import json
import os
from census import Census
from us import states

try:
    from uszipcode import SearchEngine
    USZIPCODE_AVAILABLE = True
except ImportError:
    USZIPCODE_AVAILABLE = False
    print("Warning: uszipcode package not available. Install with: pip install uszipcode")

# Cache demographic data to avoid repeated API calls
ZIP_DEMOGRAPHICS_CACHE = {}

# Initialize ZIP search engine (singleton)
_zip_search_engine = None

def get_zip_search_engine():
    """Get or create the ZIP code search engine singleton."""
    global _zip_search_engine
    if _zip_search_engine is None and USZIPCODE_AVAILABLE:
        _zip_search_engine = SearchEngine()
    return _zip_search_engine

def load_cached_demographics():
    """Load cached demographic data from file if it exists."""
    cache_file = 'zip_demographics_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_cached_demographics(data: Dict):
    """Save demographic data to cache file."""
    cache_file = 'zip_demographics_cache.json'
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def get_zip_demographics(zip_code: str) -> Dict:
    """
    Get demographic data for a ZIP code using Census API.
    Returns population, density, and income data.
    Uses caching to minimize API calls.
    """
    # Check cache first
    global ZIP_DEMOGRAPHICS_CACHE
    if not ZIP_DEMOGRAPHICS_CACHE:
        ZIP_DEMOGRAPHICS_CACHE = load_cached_demographics()
    
    if zip_code in ZIP_DEMOGRAPHICS_CACHE:
        return ZIP_DEMOGRAPHICS_CACHE[zip_code]
    
    # Initialize Census API
    c = Census('b059abd908b4797144c8ff378864f19f4b72b260',2020)
    
    try:
        print(f"Attempting to fetch Census data for ZIP {zip_code}...")
        
        # Query ACS 5-year data with enhanced variables
        results = c.acs5.get(
            ('B01003_001E',  # Total population
             'B19013_001E',  # Median household income
             'B25010_001E',  # Average household size
             'B19001_001E',  # Household income total
             'B23025_005E',  # Unemployment
             'B15003_022E',  # Bachelor's degree
             'B15003_023E',  # Master's degree
             'B15003_024E',  # Professional degree
             'B15003_025E',  # Doctorate degree
             'B11005_002E',  # Households with children
             'B25064_001E',  # Median gross rent
             'B25077_001E'), # Median home value
            {'for': f'zip code tabulation area:{zip_code}'}
        )
        print(f"Raw Census API response: {results}")
        
        if not results:
            raise ValueError(f"No data found for ZIP code {zip_code}")
            
        result = results[0]
        print(f"First result: {result}")

        # Helper function to safely convert Census values to float
        def safe_float(value, default=0):
            try:
                return float(value) if value not in [None, 'None'] else default
            except (ValueError, TypeError):
                return default

        # Extract basic demographic values
        total_pop = safe_float(result['B01003_001E'])
        median_income = safe_float(result['B19013_001E'])
        avg_household_size = safe_float(result.get('B25010_001E'))
        if np.isnan(avg_household_size):
            avg_household_size = 2.5
        
        # Extract additional demographic indicators
        unemployment = safe_float(result['B23025_005E'])
        # Sum up higher education (bachelor's and above)
        higher_ed = sum(safe_float(result[f]) for f in 
                       ['B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E'])
        households_with_children = safe_float(result['B11005_002E'])
        median_rent = safe_float(result['B25064_001E'])
        median_home_value = safe_float(result['B25077_001E'])
        
        # Get land area from uszipcode (more accurate than estimate)
        estimated_area_sqmi = 87.0  # Default fallback
        zip_city = None
        zip_county = None
        
        if USZIPCODE_AVAILABLE:
            search = get_zip_search_engine()
            if search:
                try:
                    zipcode_obj = search.by_zipcode(zip_code)
                    if zipcode_obj:
                        # Get land area in square miles
                        if zipcode_obj.land_area_in_sqmi and zipcode_obj.land_area_in_sqmi > 0:
                            estimated_area_sqmi = float(zipcode_obj.land_area_in_sqmi)
                        # Get city and county
                        if zipcode_obj.major_city:
                            zip_city = zipcode_obj.major_city
                        if zipcode_obj.county:
                            zip_county = zipcode_obj.county
                except Exception as e:
                    print(f"Warning: Could not get ZIP metadata from uszipcode for {zip_code}: {e}")
        
        # Calculate derived metrics
        population_density = total_pop / estimated_area_sqmi if estimated_area_sqmi > 0 else 0
        unemployment_rate = (unemployment / total_pop * 100) if total_pop > 0 else 0
        higher_ed_rate = (higher_ed / total_pop * 100) if total_pop > 0 else 0
        
        demo_data = {
            'total_population': total_pop,
            'population_density': population_density,
            'median_household_income': median_income,
            'land_area_sqmi': estimated_area_sqmi,
            'avg_household_size': avg_household_size,
            'unemployment_rate': unemployment_rate,
            'higher_education_rate': higher_ed_rate,
            'households_with_children': households_with_children,
            'median_rent': median_rent,
            'median_home_value': median_home_value,
            # Add some derived features
            'affordability_ratio': median_income / median_home_value if median_home_value > 0 else 0,
            'rent_to_income_ratio': (median_rent * 12) / median_income if median_income > 0 else 0,
            # Add geographic metadata from uszipcode
            'city': zip_city,
            'county': zip_county
        }
        
    except Exception as e:
        print(f"Warning: Could not fetch census data for ZIP {zip_code}: {str(e)}")
        # Fallback to estimated data without caching
        demo_data = {
            'total_population': 25000,
            'population_density': 3000,
            'median_household_income': 65000,
            'land_area_sqmi': 87.0,
            'is_estimated': True  # Flag to indicate this is estimated data
        }
        return demo_data
    
    # Only cache the results if they're real Census data
    demo_data['is_estimated'] = False  # Flag to indicate this is real data
    ZIP_DEMOGRAPHICS_CACHE[zip_code] = demo_data
    save_cached_demographics(ZIP_DEMOGRAPHICS_CACHE)
    
    return demo_data

def safe_float(value: Any) -> float:
    """
    Safely convert a value to float.

    Returns np.nan when the value is missing or cannot be converted. This
    avoids silently filling missing demographic fields with 0, which can
    mislead downstream models.
    """
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def enrich_dataframe_with_demographics(df: pd.DataFrame, zip_column: str = 'Zip Code') -> pd.DataFrame:
    """
    Add demographic features to a DataFrame based on ZIP codes.
    """
    # Define all demographic columns with their types
    demographic_columns = {
        'Population': 'float64',
        'Population_Density': 'float64',
        'Median_Income': 'float64',
        'Avg_Household_Size': 'float64',
        'Unemployment_Rate': 'float64',
        'Higher_Education_Rate': 'float64',
        'Households_With_Children': 'float64',
        'Median_Rent': 'float64',
        'Median_Home_Value': 'float64',
        'Affordability_Ratio': 'float64',
        'Rent_To_Income_Ratio': 'float64'
    }
    
    # Initialize all columns with proper dtypes (use NaN for missing values)
    for col, dtype in demographic_columns.items():
        df[col] = pd.Series(np.nan, index=df.index, dtype=dtype)
    # Add a flag column indicating whether demographics are estimated/missing
    df['Demographics_Is_Estimated'] = pd.Series(False, index=df.index, dtype='bool')
    
    # Process each unique ZIP code
    for zip_code in df[zip_column].unique():
        if pd.isna(zip_code):
            continue
            
        # Get demographic data
        demographics = get_zip_demographics(str(int(zip_code)))
        
        # Update DataFrame
        mask = df[zip_column] == zip_code
        
        # Map demographic data to DataFrame columns
        demographic_mapping = {
            'Population': 'total_population',
            'Population_Density': 'population_density',
            'Median_Income': 'median_household_income',
            'Avg_Household_Size': 'avg_household_size',
            'Unemployment_Rate': 'unemployment_rate',
            'Higher_Education_Rate': 'higher_education_rate',
            'Households_With_Children': 'households_with_children',
            'Median_Rent': 'median_rent',
            'Median_Home_Value': 'median_home_value',
            'Affordability_Ratio': 'affordability_ratio',
            'Rent_To_Income_Ratio': 'rent_to_income_ratio'
        }
        
        for df_col, demo_key in demographic_mapping.items():
            raw_value = demo_key(demographics) if callable(demo_key) else demographics.get(demo_key, None)
            df.loc[mask, df_col] = safe_float(raw_value)
        # Mark if this ZIP's demographics were estimated or not
        is_est = bool(demographics.get('is_estimated', False))
        df.loc[mask, 'Demographics_Is_Estimated'] = is_est
    
    return df