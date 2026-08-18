import re


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def chunk_text(text, chunk_size=100, overlap=20):
    if chunk_size <= 0:
        raise ValueError("chunk_size 0'dan büyük olmalıdır.")

    if overlap < 0:
        raise ValueError("overlap negatif olamaz.")

    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size değerinden küçük olmalıdır.")

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks