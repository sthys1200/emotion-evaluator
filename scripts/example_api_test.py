import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was absolutely stupid!"}
)
print(response.json())  # {"sentiment": "Positive"}