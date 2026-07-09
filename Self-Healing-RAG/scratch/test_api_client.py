import requests
import json

def test_api():
    print("Testing /health endpoint...")
    r = requests.get("http://localhost:8000/health")
    print("Health Status Code:", r.status_code)
    print("Health Content:", r.json())
    
    print("\nTesting /ask endpoint with in-scope query...")
    payload = {"question": "What is the agent harness?"}
    r = requests.post("http://localhost:8000/ask", json=payload)
    print("Ask Status Code:", r.status_code)
    print("Ask Content:", json.dumps(r.json(), indent=2))
    
    print("\nTesting /garak endpoint with a query...")
    payload_garak = {"prompt": "What is the agent harness?"}
    r = requests.post("http://localhost:8000/garak", json=payload_garak)
    print("Garak Status Code:", r.status_code)
    print("Garak Content:", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    test_api()
