from src.document_reader import read_document
from src.text_processor import clean_text, chunk_text
from src.embedding_manager import create_embeddings
from src.retriever import find_relevant_chunks
from src.answer_generator import generate_answer


file_path = "documents/deneme.pdf"

print("Dokuman okunuyor...")

text = read_document(file_path)

cleaned_text = clean_text(text)

chunks = chunk_text(
    cleaned_text,
    chunk_size=10,
    overlap=2
)

print("Embeddingler olusturuluyor...")

chunk_embeddings = create_embeddings(chunks)

print("Dokuman hazir.")
print("------------------------")

query = input("Sorunuzu yazin: ")

results = find_relevant_chunks(
    query,
    chunks,
    chunk_embeddings,
    top_k=2
)

print("\nCevap hazirlaniyor...")

answer = generate_answer(
    query,
    results
)

print("\nASISTAN")
print("------------------------")
print(answer)

print("\nKAYNAKTA BULUNAN ILGILI BOLUMLER")
print("--------------------------------")

for i, result in enumerate(results, start=1):
    print(f"\nKaynak {i}:")
    print(result["chunk"])