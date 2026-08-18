import os
from openai import OpenAI


def generate_answer(query, relevant_chunks, api_key=None):
    context = "\n\n".join(
        [result["chunk"] for result in relevant_chunks]
    )

    key = api_key or os.getenv("OPENAI_API_KEY")

    if not key:
        raise ValueError("OpenAI API anahtari bulunamadi.")

    client = OpenAI(api_key=key)

    instructions = """
Sen dokuman tabanli bir yapay zeka asistanisin.

Kurallar:
- Yalnizca sana verilen dokuman baglamindaki bilgileri kullan.
- Dokumanda bulunmayan bilgileri uydurma.
- Cevabi Turkce, kisa ve anlasilir ver.
- Eger cevap baglamda yoksa:
  "Bu bilgi yuklenen dokumanda bulunmuyor."
  cevabini ver.
"""

    prompt = f"""
DOKUMAN BAGLAMI:
{context}

KULLANICI SORUSU:
{query}

Soruyu sadece yukaridaki dokuman baglamina gore cevapla.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=instructions,
        input=prompt
    )

    return response.output_text