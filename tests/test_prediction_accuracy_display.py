"""The range and explanation shown with a prediction must come from the model that made it."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

import source.web_app as web_app
from source.auth import SessionManager
from source.franchise_db import FranchiseDatabase
from franchise_model_training import error_ratio_interval

FRANCHISE = 'f1'


@pytest.fixture
def db(tmp_path):
    database = FranchiseDatabase(str(tmp_path / 'test.db'))
    database.create_franchise(FRANCHISE, 'Test Franchise', 'f1@example.com', 'hash')
    with patch.object(web_app, 'franchise_db', database):
        yield database


def _record_outcome(db, i, predicted, actual, franchise_id=FRANCHISE):
    db.record_prediction(
        prediction_id=f'{franchise_id}-{i}', franchise_id=franchise_id, model_id='clean_layers_ensemble',
        event_name=f'Event {i}', duration_hours=2.0, predicted_revenue_per_hour=predicted / 2.0,
        predicted_total_revenue=predicted, confidence_lower=0.0, confidence_upper=0.0,
        actual_total_net_sales=actual, event_status='completed',
    )


def test_error_ratio_interval_is_the_central_share_of_outcomes():
    predicted = np.full(101, 100.0)
    actual = np.arange(101, dtype=float)  # ratios 0.00 .. 1.00

    interval = error_ratio_interval(actual, predicted, coverage=0.8)

    assert interval['lower'] == pytest.approx(0.10)
    assert interval['median'] == pytest.approx(0.50)
    assert interval['upper'] == pytest.approx(0.90)
    assert error_ratio_interval([100.0], [0.0]) is None  # nothing usable


def test_ensemble_range_comes_from_its_real_track_record(db):
    # This model consistently lands at half of what it predicts.
    for i in range(40):
        _record_outcome(db, i, predicted=1000.0, actual=500.0)

    profile = web_app.get_accuracy_profile(FRANCHISE, 'merged_ensemble')

    assert profile['sample_size'] == 40
    assert profile['lower'] == pytest.approx(0.5) and profile['upper'] == pytest.approx(0.5)
    assert profile['mae_per_hour'] == pytest.approx(250.0)
    notes = ' '.join(web_app.generate_confidence_explanation(profile, 'merged_ensemble'))
    assert 'run high' in notes and '50%' in notes


def test_unmeasured_model_says_so_instead_of_claiming_confidence(db):
    profile = web_app.get_accuracy_profile(FRANCHISE, 'merged_ensemble')

    assert profile['sample_size'] == 0 and profile['mae_per_hour'] is None
    assert (profile['lower'], profile['upper']) == web_app.DEFAULT_INTERVAL_RATIOS
    assert 'Not enough completed events' in ' '.join(
        web_app.generate_confidence_explanation(profile, 'merged_ensemble'))


def _train_tiny_franchise_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'signal': rng.uniform(0, 10, 200), 'noise': rng.uniform(0, 10, 200)})
    model = XGBRegressor(n_estimators=30, max_depth=3, random_state=0, n_jobs=1)
    model.fit(X, 100.0 * X['signal'])
    return model


def test_franchise_explanation_adds_up_to_the_models_own_prediction():
    model = _train_tiny_franchise_model()
    X = pd.DataFrame({'signal': [9.0], 'noise': [1.0]})

    factors = web_app.explain_franchise_prediction(X, model)

    assert factors[0]['feature'] == 'signal'
    assert factors[0]['explanation'].startswith('Raises')
    contribs = model.get_booster().predict(__import__('xgboost').DMatrix(X), pred_contribs=True)[0]
    assert contribs.sum() == pytest.approx(float(model.predict(X)[0]), rel=1e-4)


def test_predict_route_reports_franchise_models_own_range_and_factors(db):
    model = _train_tiny_franchise_model()
    db.upsert_model(
        model_id=f'franchise_{FRANCHISE}', franchise_id=FRANCHISE, model_name='Your Franchise Model',
        model_path='unused', model_type='franchise_specific', cv_mae=42.0, data_records_count=123,
        training_metadata_json=json.dumps({'metrics': {
            'cv_mae_per_hour': 42.0,
            'interval_lower_ratio': 0.6, 'interval_median_ratio': 1.0, 'interval_upper_ratio': 1.5,
        }}),
    )

    def fake_features(form_data, franchise_id, demographics_dict, weather_dict,
                      franchise_history=None, return_raw_values=False):
        return pd.DataFrame([{'signal': 9.0, 'noise': 1.0}]), {'signal': 'nine'}

    client = web_app.app.test_client()
    client.set_cookie(SessionManager.get_session_cookie_name(), 'token')
    with patch.object(web_app.auth_manager, 'validate_session', return_value=(FRANCHISE, None)), \
         patch.object(web_app, 'PROD_MANAGER', object()), \
         patch.object(web_app, 'get_franchise_serving_model', return_value=(model, [])), \
         patch.object(web_app, 'get_demographics_from_zip', return_value={}), \
         patch.object(web_app, 'get_weather_for_event', return_value={}), \
         patch('clean_layers_feature_preparation.prepare_features_for_clean_layers_model', fake_features):
        response = client.post('/predict', data={
            'event_date': '2026-10-10', 'duration': '2', 'zip_code': '60540', 'event_name': 'Fall Fest',
        })

    data = response.get_json()
    assert data['success'], data
    assert data['model_type'] == 'franchise_specific'
    total = data['total_net_sales']
    assert data['range_lower'] == pytest.approx(total * 0.6)
    assert data['range_upper'] == pytest.approx(total * 1.5)
    assert data['confidence_percentage'] == 80
    assert data['model_cv_mae'] == 42.0 and data['accuracy_sample_size'] == 123
    assert data['feature_contributions'][0]['feature'] == 'signal'
    assert data['feature_contributions'][0]['value'] == 'nine'
    assert data['advanced_explanation'] is None  # that panel describes the ensemble
    assert any('franchise model' in note for note in data['confidence_explanations'])
