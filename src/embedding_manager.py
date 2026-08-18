from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_model():
    global _model

    if _model is None:
        print("Embedding modeli yükleniyor...")
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def create_embeddings(texts):
    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    return embeddings