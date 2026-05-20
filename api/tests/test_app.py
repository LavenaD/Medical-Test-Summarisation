import sys
from pathlib import Path

from fastapi.testclient import TestClient
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

def test_summarise_returns_summary():
    payload = {"medical_text": "heart size is normal the lungs are clear no pneumothorax or pleural effusion"}
    expected_summary = "heart size normal lungs clear no pneumothorax no pleural effusion"
    with patch("app.run_inference", return_value=expected_summary) as mock_run_inference:
        response = client.post("/summarize", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "input": payload["medical_text"],
            "summary": expected_summary
        }
        mock_run_inference.assert_called_once_with(payload["medical_text"])

