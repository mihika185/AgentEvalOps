from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint_is_available():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)
    assert "status" in data


def test_openapi_schema_includes_core_routes():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    paths = schema["paths"]

    assert "/api/v1/documents/upload-and-index" in paths
    assert "/api/v1/rag/answer" in paths
    assert "/api/v1/benchmarks/datasets" in paths
    assert "/api/v1/quality-gates" in paths