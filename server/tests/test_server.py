import pytest
from server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_image(client):
    # اینجا می‌توانید یک تست برای آپلود تصویر بنویسید
    response = client.post('/upload', json={
        "image": "base64_encoded_image_here",
        "nationalCode": "1234567890",
        "firstName": "John",
        "lastName": "Doe"
    })
    assert response.status_code == 200
    assert b"success" in response.data
