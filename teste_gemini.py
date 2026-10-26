
import os
from google import genai
from google.genai import types

cliente = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),
)

modelo = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print(f"Testando modelo: {modelo}")

resposta = cliente.models.generate_content(
    model=modelo,
    contents="Responda apenas: OK",
)

print(resposta.text)
