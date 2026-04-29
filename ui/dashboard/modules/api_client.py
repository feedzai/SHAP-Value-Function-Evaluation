# modules/api_client.py
import requests

API_BASE = "http://localhost:8000"


def fetch_datasets():
    """Fetches all datasets and returns a dictionary."""
    try:
        datasets = requests.get(f"{API_BASE}/datasets").json()
        return {d["name"]: d["dataset_id"] for d in datasets}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching datasets: {e}")
        return {}


def fetch_random_case(dataset_id: str):
    """Fetches a random case for a given dataset ID."""
    try:
        response = requests.get(f"{API_BASE}/case/{dataset_id}").json()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching case, dataset_id: {dataset_id}: {e}")
        return None


def fetch_background_cases(dataset_id: str, size: int = 1):
    """Fetches background cases for a given dataset ID."""
    try:
        response = requests.get(
            f"{API_BASE}/background/{dataset_id}", params={"size": size}
        ).json()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching background instance, dataset_id: {dataset_id}: {e}")
        return None


def submit_case_response(case_response: dict):
    """Submits a case response to the API."""
    try:
        response = requests.post(f"{API_BASE}/case-response", json=case_response)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error submitting case response: {e}")
        return None
