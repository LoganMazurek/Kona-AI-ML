import json

from franchise_model_training import FranchiseModelTrainer


def _ex(name, date, total, hours=2.0):
    return {
        'event_name': name,
        'scheduled_event_date': date,
        'actual_total_net_sales': total,
        'duration_hours': hours,
        'event_features_json': json.dumps({'Temp': 80.0}),
    }


def test_training_frame_drops_low_revenue_and_uses_only_prior_history(tmp_path):
    trainer = FranchiseModelTrainer(franchise_db=None, models_dir=str(tmp_path))
    df = trainer._build_training_frame([
        _ex('Fall Fest', '2025-09-01', 600),     # $300/hr, first occurrence
        _ex('fall  FEST', '2025-09-01', 800),    # same day: must not see sibling
        _ex('Fall Fest', '2025-09-02', 40),      # $20/hr washout: dropped
        _ex('Fall Fest', '2026-09-01', 1000),    # sees the two 2025-09-01 runs
        _ex('Other', '2026-09-02', 500, hours=0),  # zero duration: dropped
    ])

    assert len(df) == 3
    assert list(df['Prior_Event_Count']) == [0, 0, 2]
    assert list(df['Prior_Event_Revenue_Per_Hour']) == [0, 0, 350]


def test_franchise_model_gets_history_features_in_trained_column_order():
    import pandas as pd
    import source.web_app as web_app

    class FakeModel:
        feature_names_in_ = ['Prior_Event_Revenue_Per_Hour', 'Temp', 'Prior_Event_Count']

        def predict(self, X):
            assert list(X.columns) == self.feature_names_in_
            return [X.iloc[0]['Prior_Event_Revenue_Per_Hour'] * 10]

    history = [_ex('Fall Fest', '2025-09-01', 600), _ex('Fall Fest', '2027-01-01', 9999)]
    feature_df = pd.DataFrame([{'Temp': 80.0, 'Unused': 1.0}])

    preds, model_type = web_app.predict_event_revenue(
        feature_df, 'f1', FakeModel(), history, ' fall fest ', '2026-09-01')

    assert model_type == 'franchise_specific'
    assert preds[0] == 3000  # only the earlier $300/hr run counts
