import os
import tempfile
from pathlib import Path

import streamlit as st

from src.document_reader import read_document
from src.text_processor import clean_text, chunk_text
from src.embedding_manager import create_embeddings
from src.retriever import find_relevant_chunks
from src.answer_generator import generate_answer


st.set_page_config(
    page_title="Dokuman Tabanli Yapay Zeka Asistani",
    page_icon="📄",
    layout="centered"
)


st.title("📄 Dokuman Tabanli Yapay Zeka Asistani")

st.write(
    "PDF veya TXT dosyanizi yukleyin ve dokuman hakkinda sorular sorun."
)


uploaded_file = st.file_uploader(
    "Dokuman yukleyin",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    st.success(f"Dosya yuklendi: {uploaded_file.name}")

    suffix = Path(uploaded_file.name).suffix

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        with st.spinner("Dokuman hazirlaniyor..."):

            text = read_document(temp_path)

            cleaned_text = clean_text(text)

            if not cleaned_text:
                st.error(
                    "Bu dokumandan okunabilir metin bulunamadi."
                )
                st.stop()

            chunks = chunk_text(
                cleaned_text,
                chunk_size=100,
                overlap=20
            )

            chunk_embeddings = create_embeddings(chunks)

        st.success(
            f"Dokuman hazir. {len(chunks)} metin parcasi olusturuldu."
        )

        question = st.text_input(
            "Sorunuzu yazin",
            placeholder="Ornek: Sirket nerede bulunuyor?"
        )


        if st.button("Soruyu Cevapla"):

            if not question.strip():

                st.warning("Lutfen bir soru yazin.")

            else:

                with st.spinner("Cevap hazirlaniyor..."):

                    results = find_relevant_chunks(
                        question,
                        chunks,
                        chunk_embeddings,
                        top_k=3
                    )

                    api_key = os.getenv("OPENAI_API_KEY")

                    if not api_key:
                        try:
                            api_key = st.secrets["OPENAI_API_KEY"]
                        except Exception:
                            api_key = None

                    if not api_key:

                        st.error(
                            "OpenAI API anahtari bulunamadi."
                        )

                    else:

                        try:

                            answer = generate_answer(
                                question,
                                results,
                                api_key=api_key
                            )

                            st.subheader("🤖 Asistan")

                            st.write(answer)

                            st.caption(
                                f"Kaynak dokuman: {uploaded_file.name}"
                            )

                            with st.expander(
                                "Kullanilan dokuman bolumlerini goster"
                            ):

                                for i, result in enumerate(
                                    results,
                                    start=1
                                ):

                                    st.markdown(
                                        f"**Kaynak {i} "
                                        f"(Benzerlik: "
                                        f"{result['score']:.3f})**"
                                    )

                                    st.write(
                                        result["chunk"]
                                    )

                        except Exception as error:

                            st.error(
                                "Cevap olusturulurken "
                                "API hatasi meydana geldi."
                            )

                            st.code(str(error))

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)