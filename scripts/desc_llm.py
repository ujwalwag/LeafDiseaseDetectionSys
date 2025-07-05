import requests
import json


GEMINI_API_KEY = "AIzaSyB0vHx2Aphx3bJ_iny_sUS1EvOYOpu9IZ4" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def generate_llm_description(predicted_class_label):
    """
    Generates a concise, scientific description for the predicted disease label using the Gemini API.
    """
    prompt = f"Provide a concise, scientific description of the plant disease: {predicted_class_label}. Focus on symptoms and common characteristics."
    
    headers = {
        'Content-Type': 'application/json'
    }
    api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}" if GEMINI_API_KEY else GEMINI_API_URL

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            return generated_text
        else:
            print(f"LLM response missing expected structure: {result}")
            return "Failed to generate description (LLM response format error)."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"Failed to generate description (API error: {e})."
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini API: {e}")
        return "Failed to generate description (JSON decode error)."
    except Exception as e:
        print(f"Unexpected error in LLM description generation: {e}")
        return "Failed to generate description (unexpected error)."

if __name__ == "__main__":

    print("Testing LLM description generation...")
    test_label = "Tomato___Bacterial_spot"
    description = generate_llm_description(test_label)
    print(f"\nDescription for '{test_label}':\n{description}")

    test_label_healthy = "Apple___healthy"
    description_healthy = generate_llm_description(test_label_healthy)
    print(f"\nDescription for '{test_label_healthy}':\n{description_healthy}")