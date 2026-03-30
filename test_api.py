# test_api.py
import requests
import json

# Test data
test_data = {
    'pregnancies': 2,
    'glucose': 120,
    'bloodPressure': 70,
    'skinThickness': 20,
    'insulin': 80,
    'bmi': 25.5,
    'dpf': 0.5,
    'age': 30
}

# Test health endpoint
print("Testing health endpoint...")
response = requests.get('http://localhost:5000/api/health')
print(f"Health: {response.json()}")

# Test prediction endpoint
print("\nTesting prediction endpoint...")
response = requests.post('http://localhost:5000/api/predict', json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test model performance endpoint
print("\nTesting model performance endpoint...")
response = requests.get('http://localhost:5000/api/model-performance')
print(f"Performance: {json.dumps(response.json(), indent=2)}")