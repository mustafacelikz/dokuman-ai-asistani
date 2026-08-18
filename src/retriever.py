from sentence_transformers import util

from src.embedding_manager import create_embeddings


def find_relevant_chunks(query, chunks, chunk_embeddings, top_k=2):
    query_embedding = create_embeddings([query])

    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    top_results = scores.topk(
        k=min(top_k, len(chunks))
    )

    results = []

    for score, index in zip(top_results.values, top_results.indices):
        results.append(
            {
                "chunk": chunks[index.item()],
                "score": score.item()
            }
        )

    return results