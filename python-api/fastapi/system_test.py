from fastapi.testclient import TestClient
from main import app

# Create a test client using the application in main.py
client = TestClient(app)


def test_main():
    response = client.get("/items?name=scissors")
    assert response.status_code == 200
    assert response.json() == {"name": "scissors", "quantity": 100}


# pytest
