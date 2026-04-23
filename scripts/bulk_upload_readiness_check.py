"""Readiness checks for bulk upload scenarios.

This script exercises the /bulk-upload flow end-to-end using Flask's test
client with deterministic mocks for prediction/dependency calls.

Scenarios covered:
1. New future events are added with non-zero predictions.
2. Re-uploading the same sheet does not create duplicate rows.
3. Existing past events can ingest actual net sales on a later upload.
4. Mixed files (valid + duplicate/update + invalid row) behave correctly.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep startup lightweight when importing web_app.
os.environ.setdefault("ENABLE_PREDICTION_EXPLAINER", "false")
os.environ.setdefault("ENABLE_USZIPCODE", "false")

import web_app  # noqa: E402
from auth import AuthManager, SessionManager  # noqa: E402
from franchise_db import FranchiseDatabase  # noqa: E402
import clean_layers_feature_preparation as clfp  # noqa: E402


class FakeProductionManager:
    """Deterministic predictor used by readiness tests."""

    def predict(self, feature_df, franchise_id=None):  # pylint: disable=unused-argument
        # Return a stable, non-zero total net sales prediction.
        return [1200.0], "clean_layers_ensemble"


def fake_prepare_features_for_clean_layers_model(
    form_data,
    franchise_id,
    demographics_dict,
    weather_dict,
    franchise_history=None,
    return_raw_values=False,
):
    """Return a minimal feature frame that satisfies bulk inference path."""
    _ = (franchise_id, demographics_dict, weather_dict, franchise_history)
    df = pd.DataFrame(
        [
            {
                "industry": str(form_data.get("industry", "")),
                "duration": float(form_data.get("duration", 1.0)),
            }
        ]
    )
    raw = {"industry": form_data.get("industry", "")}
    return (df, raw) if return_raw_values else (df, None)


def _set_test_cookie(client, token: str) -> None:
    name = SessionManager.get_session_cookie_name()
    try:
        client.set_cookie(key=name, value=token)
    except TypeError:
        # Compatibility path for older Flask test client signature.
        client.set_cookie("localhost", name, token)


def _query_predictions(db_path: Path, franchise_id: str) -> List[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT prediction_id, event_name, scheduled_event_date,
               predicted_total_revenue, actual_total_net_sales,
               event_status, include_in_training, bulk_dedup_key
        FROM predictions
        WHERE franchise_id = ?
        ORDER BY created_timestamp ASC
        """,
        (franchise_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def _prediction_by_event(rows: List[sqlite3.Row], event_name: str) -> sqlite3.Row:
    matches = [r for r in rows if r["event_name"] == event_name]
    if not matches:
        raise AssertionError(f"Expected event '{event_name}' not found")
    return matches[0]


def _post_excel(client, rows: List[Dict[str, Any]]):
    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_excel(tmp_path, index=False)
        with tmp_path.open("rb") as fh:
            response = client.post(
                "/bulk-upload",
                data={"file": (fh, "bulk_upload.xlsx")},
                content_type="multipart/form-data",
            )
        return response
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> int:
    print("Running bulk upload readiness checks...")

    with tempfile.TemporaryDirectory(prefix="bulk-upload-check-") as temp_dir:
        db_path = Path(temp_dir) / "franchise_data_test.db"

        # Patch app globals to isolated test dependencies.
        original_db = web_app.franchise_db
        original_auth = web_app.auth_manager
        original_prod_manager = web_app.PROD_MANAGER
        original_demographics = web_app.get_demographics_from_zip
        original_weather = web_app.get_weather_for_event
        original_feature_fn = clfp.prepare_features_for_clean_layers_model

        try:
            web_app.franchise_db = FranchiseDatabase(str(db_path))
            web_app.auth_manager = AuthManager(web_app.franchise_db)
            web_app.PROD_MANAGER = FakeProductionManager()
            web_app.get_demographics_from_zip = lambda zip_code: {
                "Total_Population": 10000,
                "Median_Household_Income": 65000,
                "Unemployment_Rate": 4.2,
                "Households_With_Children": 0.35,
                "Median_Rent": 1200,
                "Median_Home_Value": 280000,
                "Average_Household_Size": 2.6,
                "source": "default",
            }
            web_app.get_weather_for_event = lambda zip_code, event_datetime, user_temperature=None: {
                "temperature_f": float(user_temperature) if user_temperature else 70.0,
                "is_raining": 0,
                "source": "mock",
            }
            clfp.prepare_features_for_clean_layers_model = fake_prepare_features_for_clean_layers_model

            web_app.app.config["TESTING"] = True

            franchise_id = "test-franchise"
            password = "test-password-123"
            password_hash = web_app.auth_manager.hash_password(password)
            created = web_app.franchise_db.create_franchise(
                franchise_id=franchise_id,
                franchise_name="Test Franchise",
                email="test@example.com",
                password_hash=password_hash,
            )
            _assert(created, "Failed to create test franchise")

            token, error = web_app.auth_manager.login(franchise_id, password)
            _assert(token is not None and error is None, f"Login failed: {error}")

            client = web_app.app.test_client()
            _set_test_cookie(client, token)

            # Scenario 1: New future events are added with predictions.
            future_rows = [
                {
                    "Event Name": "Future Event A",
                    "Industry": "Festival",
                    "Equipment Type": "Blended Truck",
                    "Equipment Number": "TRK-01",
                    "Source Event ID": "SRC-1001",
                    "Event Date": "2099-06-10",
                    "Start Time": "10:00",
                    "Duration (Hours)": "6",
                    "City": "Chicago",
                    "ZIP Code": "60601",
                    "Temperature (°F)": "78",
                    "Net Sales": "",
                },
                {
                    "Event Name": "Future Event B",
                    "Industry": "Corporate",
                    "Equipment Type": "Truck",
                    "Equipment Number": "TRK-02",
                    "Source Event ID": "SRC-1002",
                    "Event Date": "2099-06-11",
                    "Start Time": "11:00",
                    "Duration (Hours)": "4",
                    "City": "Chicago",
                    "ZIP Code": "60602",
                    "Temperature (°F)": "76",
                    "Net Sales": "",
                },
            ]
            resp = _post_excel(client, future_rows)
            _assert(resp.status_code == 200, "Scenario 1 upload failed")
            rows = _query_predictions(db_path, franchise_id)
            _assert(len(rows) == 2, f"Scenario 1 expected 2 predictions, found {len(rows)}")
            _assert(
                all(float(r["predicted_total_revenue"] or 0) > 0 for r in rows),
                "Scenario 1 expected non-zero predictions for all new events",
            )
            print("[PASS] Scenario 1: new future events are added with predictions")

            # Scenario 2: Re-upload same file does not duplicate.
            resp = _post_excel(client, future_rows)
            _assert(resp.status_code == 200, "Scenario 2 re-upload failed")
            rows = _query_predictions(db_path, franchise_id)
            _assert(len(rows) == 2, f"Scenario 2 expected still 2 predictions, found {len(rows)}")
            print("[PASS] Scenario 2: re-upload skips duplicates")

            # Scenario 3: Existing past event receives actuals on later upload.
            past_no_actual = [
                {
                    "Event Name": "Past Event A",
                    "Industry": "Festival",
                    "Equipment Type": "Truck",
                    "Equipment Number": "TRK-03",
                    "Source Event ID": "SRC-2001",
                    "Event Date": "2024-05-05",
                    "Start Time": "09:00",
                    "Duration (Hours)": "5",
                    "City": "Naperville",
                    "ZIP Code": "60540",
                    "Temperature (°F)": "72",
                    "Net Sales": "",
                }
            ]
            resp = _post_excel(client, past_no_actual)
            _assert(resp.status_code == 200, "Scenario 3 initial past upload failed")

            before_update = _query_predictions(db_path, franchise_id)
            _assert(len(before_update) == 3, "Scenario 3 expected one additional row before outcome update")
            past_row = _prediction_by_event(before_update, "Past Event A")
            _assert(past_row["actual_total_net_sales"] is None, "Scenario 3 expected no actual on first upload")

            past_with_actual = [dict(past_no_actual[0], **{"Net Sales": "1850"})]
            resp = _post_excel(client, past_with_actual)
            _assert(resp.status_code == 200, "Scenario 3 update upload failed")

            after_update = _query_predictions(db_path, franchise_id)
            _assert(len(after_update) == 3, "Scenario 3 expected no duplicate row when updating actual")
            past_updated = _prediction_by_event(after_update, "Past Event A")
            _assert(
                float(past_updated["actual_total_net_sales"] or 0) == 1850.0,
                "Scenario 3 expected actual_total_net_sales to be updated to 1850",
            )
            _assert(past_updated["event_status"] == "completed", "Scenario 3 expected status=completed")
            _assert(int(past_updated["include_in_training"] or 0) == 1, "Scenario 3 expected include_in_training=1")
            print("[PASS] Scenario 3: existing past event ingests later actuals")

            # Scenario 4: Mixed file adds new, updates existing, and rejects invalid rows.
            seed_past = [
                {
                    "Event Name": "Past Event B",
                    "Industry": "Community",
                    "Equipment Type": "Truck",
                    "Equipment Number": "TRK-05",
                    "Source Event ID": "SRC-3001",
                    "Event Date": "2024-06-15",
                    "Start Time": "12:00",
                    "Duration (Hours)": "3",
                    "City": "Aurora",
                    "ZIP Code": "60505",
                    "Temperature (°F)": "71",
                    "Net Sales": "",
                }
            ]
            resp = _post_excel(client, seed_past)
            _assert(resp.status_code == 200, "Scenario 4 seed upload failed")

            pre_mixed_rows = _query_predictions(db_path, franchise_id)
            pre_count = len(pre_mixed_rows)

            mixed_rows = [
                {
                    "Event Name": "Invalid Missing City",
                    "Industry": "Festival",
                    "Equipment Type": "Truck",
                    "Equipment Number": "TRK-06",
                    "Source Event ID": "SRC-4001",
                    "Event Date": "2099-07-01",
                    "Start Time": "10:00",
                    "Duration (Hours)": "4",
                    "City": "",
                    "ZIP Code": "60603",
                    "Temperature (°F)": "75",
                    "Net Sales": "",
                },
                {
                    "Event Name": "Past Event B",
                    "Industry": "Community",
                    "Equipment Type": "Truck",
                    "Equipment Number": "TRK-05",
                    "Source Event ID": "SRC-3001",
                    "Event Date": "2024-06-15",
                    "Start Time": "12:00",
                    "Duration (Hours)": "3",
                    "City": "Aurora",
                    "ZIP Code": "60505",
                    "Temperature (°F)": "71",
                    "Net Sales": "900",
                },
                {
                    "Event Name": "Future Event C",
                    "Industry": "Corporate",
                    "Equipment Type": "Blended Truck",
                    "Equipment Number": "TRK-07",
                    "Source Event ID": "SRC-4003",
                    "Event Date": "2099-07-03",
                    "Start Time": "13:00",
                    "Duration (Hours)": "5",
                    "City": "Chicago",
                    "ZIP Code": "60604",
                    "Temperature (°F)": "80",
                    "Net Sales": "",
                },
            ]
            resp = _post_excel(client, mixed_rows)
            _assert(resp.status_code == 200, "Scenario 4 mixed upload failed")

            post_mixed_rows = _query_predictions(db_path, franchise_id)
            _assert(
                len(post_mixed_rows) == pre_count + 1,
                "Scenario 4 expected exactly one new row (invalid rejected, existing updated)",
            )
            past_b = _prediction_by_event(post_mixed_rows, "Past Event B")
            _assert(
                float(past_b["actual_total_net_sales"] or 0) == 900.0,
                "Scenario 4 expected Past Event B outcome update to 900",
            )
            future_c = _prediction_by_event(post_mixed_rows, "Future Event C")
            _assert(
                float(future_c["predicted_total_revenue"] or 0) > 0,
                "Scenario 4 expected prediction for newly added Future Event C",
            )
            print("[PASS] Scenario 4: mixed file behavior is correct")

            print("All bulk upload readiness scenarios passed.")
            return 0

        finally:
            # Restore application globals.
            web_app.franchise_db = original_db
            web_app.auth_manager = original_auth
            web_app.PROD_MANAGER = original_prod_manager
            web_app.get_demographics_from_zip = original_demographics
            web_app.get_weather_for_event = original_weather
            clfp.prepare_features_for_clean_layers_model = original_feature_fn


if __name__ == "__main__":
    raise SystemExit(run())
