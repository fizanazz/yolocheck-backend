import httpx 
from app.core.config import get_settings 
s = get_settings() 
print('Key:', s.gemini_api_key[:25]) 
print('Model:', s.gemini_model) 
url = f'https://generativelanguage.googleapis.com/v1beta/models/{s.gemini_model}:generateContent?key={s.gemini_api_key}' 
r = httpx.post(url, json={'contents': [{'parts': [{'text': 'What is a mole?'}], 'role': 'user'}], 'generationConfig': {'maxOutputTokens': 100}}, timeout=15) 
print('Status:', r.status_code) 
print('Response:', r.text[:500]) 
