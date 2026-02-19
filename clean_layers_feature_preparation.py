"""
Updated feature preparation for clean_layers production model.

This replaces the old Phase 1/Phase 2 engineering with a simple,
clean data layer approach that matches the production model exactly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.preprocessing import LabelEncoder


def get_required_features():
    """Load the exact feature list expected by production model."""
    features_path = os.path.join(
        os.path.dirname(__file__),
        'production', 'models', 'clean_layers_feature_names.json'
    )
    
    try:
        with open(features_path) as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to hardcoded list if file not found
        return [
            "franchise_id", "Event_Industry", "equipment_type", "has_smoothie_capability",
            "Duration_Hours_x", "Total_Population", "Median_Household_Income", "Unemployment_Rate",
            "Average_Household_Size", "Rent_To_Income_Ratio", "Median_Rent", "Median_Home_Value",
            "Households_With_Children", "Population_Density", "Higher_Education_Rate",
            "Affordability_Ratio", "Day_Of_Month", "Day_Of_Week", "Duration_Bracket",
            "Duration_Hours_y", "Hour", "Hour_Bucket", "Is_Holiday_Weekend", "Is_Near_Holiday",
            "Is_Prime_Time", "Is_Weekend", "Month", "Quarter", "School_Season", "Season",
            "Week_Of_Year", "Temperature_F", "Is_Raining", "Precipitation_Inches", "Wind_Speed_MPH",
            "Cloud_Cover_Percent", "Feels_Like_Temp_F", "Industry_Avg_7D", "Industry_Avg_14D",
            "Industry_Avg_30D", "County_Avg_30D", "Competing_Events_Same_Day",
            "Competing_Events_Same_Industry", "Events_Past_30d", "Equipment_Industry", "Equipment_Season"
        ]


def prepare_features_for_clean_layers_model(
    form_data,
    franchise_id,
    demographics_dict,
    weather_dict,
    franchise_history=None,
    return_raw_values=False
):
    """
    Prepare exactly 46 features for clean_layers production model.
    
    Args:
        form_data: Dict with keys: event_date, start_time, duration, industry, equipment_type, 
                   has_smoothie_capability, zip_code, city
        franchise_id: The franchise ID (from auth session)
        demographics_dict: Dict with demographic data for zip code
        weather_dict: Dict with weather data
        franchise_history: Optional dict with historical event data
    
    Returns:
        pd.DataFrame: Single row with exactly 46 features in correct order
    """
    
    # Parse timing information
    event_date_str = form_data.get('event_date', datetime.now().strftime('%Y-%m-%d'))
    start_time_str = form_data.get('start_time', '18:00')
    
    try:
        event_date = datetime.strptime(event_date_str, '%Y-%m-%d')
        start_hour = int(start_time_str.split(':')[0])
    except (ValueError, AttributeError):
        event_date = datetime.now()
        start_hour = 18
    
    duration_hours = float(form_data.get('duration', 4.0))
    
    # ===== BUILD FEATURES (46 total) =====
    features = {}
    
    # 1. Franchise & Equipment (4 features)
    features['franchise_id'] = franchise_id
    features['Event_Industry'] = form_data.get('industry', 'Food Service')
    features['equipment_type'] = form_data.get('equipment_type', 'truck')
    features['has_smoothie_capability'] = int(form_data.get('has_smoothie_capability', 0))
    
    # 2. Duration (2 features - both versions for compatibility)
    features['Duration_Hours_x'] = duration_hours
    features['Duration_Hours_y'] = duration_hours
    
    # Duration bracket (categorize into ranges)
    if duration_hours <= 2:
        features['Duration_Bracket'] = 'Short (0-2h)'
    elif duration_hours <= 4:
        features['Duration_Bracket'] = 'Medium (2-4h)'
    elif duration_hours <= 6:
        features['Duration_Bracket'] = 'Long (4-6h)'
    else:
        features['Duration_Bracket'] = 'Extended (6h+)'
    
    # 3. Demographics (11 features)
    features['Total_Population'] = demographics_dict.get('Total_Population', 50000)
    features['Median_Household_Income'] = demographics_dict.get('Median_Household_Income', 65000)
    features['Unemployment_Rate'] = demographics_dict.get('Unemployment_Rate', 5.0)
    features['Average_Household_Size'] = demographics_dict.get('Average_Household_Size', 2.5)
    features['Rent_To_Income_Ratio'] = demographics_dict.get('Rent_To_Income_Ratio', 0.30)
    features['Median_Rent'] = demographics_dict.get('Median_Rent', 1500)
    features['Median_Home_Value'] = demographics_dict.get('Median_Home_Value', 300000)
    features['Households_With_Children'] = demographics_dict.get('Households_With_Children', 30)
    features['Population_Density'] = demographics_dict.get('Population_Density', 2000)
    features['Higher_Education_Rate'] = demographics_dict.get('Higher_Education_Rate', 30)
    features['Affordability_Ratio'] = (
        features['Median_Home_Value'] / features['Median_Household_Income']
        if features['Median_Household_Income'] > 0 else 4.6
    )
    
    # 4. Temporal/Date Features (10 features)
    features['Day_Of_Month'] = event_date.day
    features['Day_Of_Week'] = event_date.weekday()  # 0=Monday, 6=Sunday
    features['Hour'] = start_hour
    features['Is_Weekend'] = 1 if event_date.weekday() >= 5 else 0
    features['Month'] = event_date.month
    features['Quarter'] = (event_date.month - 1) // 3 + 1
    features['Week_Of_Year'] = event_date.isocalendar()[1]
    
    # Season
    month = event_date.month
    if month in [12, 1, 2]:
        features['Season'] = 'Winter'
    elif month in [3, 4, 5]:
        features['Season'] = 'Spring'
    elif month in [6, 7, 8]:
        features['Season'] = 'Summer'
    else:
        features['Season'] = 'Fall'
    
    # Hour bucket (categorize hour into time of day)
    if 6 <= start_hour < 10:
        features['Hour_Bucket'] = 'Morning'
    elif 10 <= start_hour < 14:
        features['Hour_Bucket'] = 'Midday'
    elif 14 <= start_hour < 17:
        features['Hour_Bucket'] = 'Afternoon'
    elif 17 <= start_hour < 21:
        features['Hour_Bucket'] = 'Evening'
    else:
        features['Hour_Bucket'] = 'Night'
    
    # Prime time (typically 5pm-9pm weekdays, 11am-9pm weekends)
    is_weekday = features['Day_Of_Week'] < 5
    if is_weekday:
        features['Is_Prime_Time'] = 1 if 17 <= start_hour < 21 else 0
    else:
        features['Is_Prime_Time'] = 1 if 11 <= start_hour < 21 else 0
    
    # School season (typically September-May)
    features['School_Season'] = 1 if month in [9, 10, 11, 12, 1, 2, 3, 4, 5] else 0
    
    # Holiday/near-holiday detection
    features['Is_Holiday_Weekend'] = _is_holiday_or_near_holiday(event_date)
    features['Is_Near_Holiday'] = _is_near_holiday(event_date)
    
    # 5. Weather Features (6 features)
    features['Temperature_F'] = weather_dict.get('temperature_f', 70.0)
    features['Is_Raining'] = int(weather_dict.get('is_raining', 0))
    features['Precipitation_Inches'] = weather_dict.get('precipitation_inches', 0.0)
    features['Wind_Speed_MPH'] = weather_dict.get('wind_speed_mph', 8.0)
    features['Cloud_Cover_Percent'] = weather_dict.get('cloud_cover_percent', 50)
    features['Feels_Like_Temp_F'] = weather_dict.get('feels_like_temp_f', 70.0)
    
    # 6. Historical Performance Features (7 features - use defaults if franchise_history not available)
    if franchise_history:
        features['Industry_Avg_7D'] = franchise_history.get('industry_avg_7d', 155.0)
        features['Industry_Avg_14D'] = franchise_history.get('industry_avg_14d', 157.0)
        features['Industry_Avg_30D'] = franchise_history.get('industry_avg_30d', 155.0)
        features['County_Avg_30D'] = franchise_history.get('county_avg_30d', 175.0)
        features['Events_Past_30d'] = float(franchise_history.get('events_past_30d', 10))
        features['Competing_Events_Same_Day'] = float(franchise_history.get('competing_same_day', 1))
        features['Competing_Events_Same_Industry'] = float(franchise_history.get('competing_same_industry', 0))
    else:
        # Use reasonable defaults from training data
        features['Industry_Avg_7D'] = 155.0
        features['Industry_Avg_14D'] = 157.0
        features['Industry_Avg_30D'] = 155.0
        features['County_Avg_30D'] = 175.0
        features['Events_Past_30d'] = 10.0
        features['Competing_Events_Same_Day'] = 1.0
        features['Competing_Events_Same_Industry'] = 0.0
    
    # 7. Interaction Features (2 features)
    features['Equipment_Industry'] = f"{features['equipment_type']}_{features['Event_Industry']}"
    features['Equipment_Season'] = f"{features['equipment_type']}_{features['Season']}"
    
    # ===== CREATE DATAFRAME WITH CORRECT ORDER =====
    required_features = get_required_features()
    
    # Build row with only required features in exact order
    df = pd.DataFrame([{
        feat: features.get(feat, 0) 
        for feat in required_features
    }])
    
    # Verify we have exactly 46 features
    if len(df.columns) != 46:
        raise ValueError(f"Expected 46 features, got {len(df.columns)}: {list(df.columns)}")
    
    # Define categorical and numeric columns
    categorical_columns = [
        'Event_Industry', 'equipment_type', 'Duration_Bracket', 
        'Hour_Bucket', 'Season', 'School_Season',
        'Equipment_Industry', 'Equipment_Season'  # Interaction features (also categorical)
    ]
    numeric_columns = [col for col in df.columns if col not in categorical_columns]

    raw_categorical_values = {}
    for col in categorical_columns:
        if col in df.columns:
            raw_value = features.get(col)
            if col == 'School_Season':
                raw_value = 'In session' if int(raw_value or 0) == 1 else 'Out of session'
            raw_categorical_values[col] = raw_value
    
    # Step 1: Fill NaN values BEFORE encoding
    # For categorical: use 'unknown'
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # For numeric: convert to numeric first, then fill NaN with 0
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Step 2: Map categorical columns using fixed mapping(converts to int64)
    # CatBoost requires integers for categorical features
    # Load categorical mappings from file
    mappings_path = os.path.join(
        os.path.dirname(__file__),
        'production', 'models', 'categorical_mappings.json'
    )
    try:
        with open(mappings_path) as f:
            categorical_mappings = json.load(f)['mappings']
    except FileNotFoundError:
        raise FileNotFoundError(f"Categorical mappings file not found: {mappings_path}")
    
    # Apply fixed mappings to categorical columns
    for col in categorical_columns:
        if col in df.columns:
            if col in categorical_mappings:
                # Used fixed mapping
                col_str = df[col].astype(str)
                df[col] = col_str.map(categorical_mappings[col])
                # Handle unmapped values by assigning to unknown category if it exists
                if df[col].isna().any():
                    unknown_value = categorical_mappings[col].get('unknown', 0)
                    if unknown_value is not None:
                        df[col] = df[col].fillna(unknown_value)
                    else:
                        # If no unknown category, fill with -1
                        df[col] = df[col].fillna(-1).astype('int64')
                df[col] = df[col].astype('int64')
            else:
                # For interactive features, use labelencoder as fallback
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                df[col] = pd.Series(encoded, index=df.index, dtype='int64')
    
    # Step 3: Final validation - ensure all columns have proper dtypes
    # Categorical columns must be integer or object (string), not float
    for col in categorical_columns:
        if col in df.columns and df[col].dtype == 'float64':
            # Convert float to int
            df[col] = df[col].astype('int64')
    
    # Numeric columns should be float64
    for col in numeric_columns:
        if col in df.columns and df[col].dtype != 'float64':
            try:
                df[col] = df[col].astype('float64')
            except:
                # If conversion fails, try numeric conversion first
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    print(f"✓ Prepared {len(df.columns)} features for clean_layers model")
    print(f"  Columns: {list(df.columns)[:5]}... (showing first 5)")
    print(f"  Encoded {len(categorical_columns)} categorical columns using fixed mappings")
    
    if return_raw_values:
        return df, raw_categorical_values
    return df


def _is_holiday_or_near_holiday(date_obj):
    """Check if date is a holiday or holiday weekend."""
    # Major U.S. holidays and surrounding dates
    month_day = (date_obj.month, date_obj.day)
    
    holiday_dates = {
        (1, 1),   # New Year's
        (1, 21),  # MLK Day (3rd Monday)
        (2, 18),  # Presidents Day (3rd Monday)
        (3, 17),  # St. Patrick's
        (5, 27),  # Memorial Day (last Monday)
        (7, 4),   # Independence Day
        (9, 2),   # Labor Day (1st Monday)
        (10, 14), # Columbus Day (2nd Monday)
        (11, 11), # Veterans Day
        (11, 28), # Thanksgiving (4th Thursday)
        (12, 25), # Christmas
        (12, 31), # New Year's Eve
    }
    
    # Check if date is a holiday or within 2 days
    is_holiday = month_day in holiday_dates
    
    # Check nearby dates (±1 day)
    nearby_holiday = any(
        (month, day) in holiday_dates 
        for month, day in [
            (date_obj.month, date_obj.day - 1),
            (date_obj.month, date_obj.day + 1),
        ]
        if day > 0 and day <= 31
    )
    
    return 1 if (is_holiday or nearby_holiday or date_obj.weekday() >= 5) else 0


def _is_near_holiday(date_obj):
    """Check if date is within a week of a major holiday."""
    month_day = (date_obj.month, date_obj.day)
    
    holiday_dates = {
        (11, 28),  # Thanksgiving
        (12, 25),  # Christmas
        (1, 1),    # New Year's
        (7, 4),    # Independence Day
    }
    
    # Check if within 7 days of any major holiday
    for hol_month, hol_day in holiday_dates:
        holiday_date = datetime(date_obj.year, hol_month, hol_day)
        days_until = (holiday_date - date_obj).days
        if -3 <= days_until <= 3:
            return 1
    
    return 0


if __name__ == '__main__':
    # Test example
    test_form_data = {
        'event_date': '2026-06-21',
        'start_time': '18:00',
        'duration': '4',
        'industry': 'Food Service',
        'equipment_type': 'truck',
        'has_smoothie_capability': '1',
        'zip_code': '60487',
    }
    
    test_demographics = {
        'Total_Population': 50000,
        'Median_Household_Income': 65000,
        'Unemployment_Rate': 4.5,
        'Average_Household_Size': 2.6,
        'Rent_To_Income_Ratio': 0.28,
        'Median_Rent': 1400,
        'Median_Home_Value': 320000,
        'Households_With_Children': 32,
        'Population_Density': 2500,
        'Higher_Education_Rate': 35,
    }
    
    test_weather = {
        'temperature_f': 75.0,
        'is_raining': 0,
        'precipitation_inches': 0.0,
        'wind_speed_mph': 6.0,
        'cloud_cover_percent': 20,
        'feels_like_temp_f': 76.0,
    }
    
    df, raw_values = prepare_features_for_clean_layers_model(
        test_form_data,
        franchise_id=1,
        demographics_dict=test_demographics,
        weather_dict=test_weather,
        return_raw_values=True
    )
    
    print(f"\nResult DataFrame:\n{df.T}")
    print(f"\nShape: {df.shape}")
    print(f"Features: {len(df.columns)}")
