"""ZIP cluster prediction helper.

Clusters are derived from the first 3 digits of a ZIP (SCF level),
matching training-time logic in `cluster_zipcodes`.
"""


def _normalize_zip(zip_code: str) -> str:
    digits = "".join(ch for ch in str(zip_code or "") if ch.isdigit())
    if len(digits) < 3:
        return ""
    return digits.zfill(5)


def predict_zip_cluster(zip_code: str) -> str:
    """Predict ZIP cluster for a given ZIP code.

    Args:
        zip_code: ZIP code string

    Returns:
        3-digit cluster string ("000" if invalid)
    """
    normalized = _normalize_zip(zip_code)
    if not normalized:
        return "000"
    return normalized[:3]
