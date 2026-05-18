import requests
base_url = "http://127.0.0.1:8000"

response = requests.get(f"{base_url}/")
print("API Status:", response.json())

response = requests.post(f"{base_url}/summarize", json={"medical_text": "heart size is normal the lungs are clear no pneumothorax or pleural effusion"})
print("Summary Response:", response.json())