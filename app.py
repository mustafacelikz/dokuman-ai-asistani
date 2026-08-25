import os
import tempfile
from pathlib import Path

import streamlit as st

from src.document_reader import read_document
from src.text_processor import clean_text, chunk_text
from src.embedding_manager import create_embeddings
from src.retriever import find_relevant_chunks
from src.answer_generator import generate_answer


# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="Doküman AI Asistanı",
    page_icon="📄",
    layout="centered"
)


# --------------------------------------------------
# TASARIM
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, rgba(63,94,251,0.12), transparent 35%),
        radial-gradient(circle at top right, rgba(118,75,162,0.10), transparent 30%),
        #0e1117;
}

.block-container {
    max-width: 900px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 14px;
}

.hero-subtitle {
    font-size: 17px;
    color: #aab2c0;
    margin-bottom: 30px;
}

.steps {
    display: flex;
    gap: 10px;
    margin-bottom: 30px;
    flex-wrap: wrap;
}

.step {
    padding: 9px 15px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #d8dee9;
    font-size: 14px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 8px;
}

.answer-card {
    background: linear-gradient(
        135deg,
        rgba(76, 110, 245, 0.16),
        rgba(128, 90, 213, 0.10)
    );
    border: 1px solid rgba(129, 140, 248, 0.30);
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
    margin-bottom: 18px;
}

.answer-label {
    color: #9ba8ff;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}

.answer-text {
    font-size: 20px;
    line-height: 1.55;
}

.footer {
    text-align: center;
    margin-top: 55px;
    color: #737b8c;
    font-size: 13px;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(129,140,248,0.65);
    border-radius: 16px;
    background: rgba(255,255,255,0.025);
    padding: 14px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
    border: none;
    background: linear-gradient(
        90deg,
        #5865f2,
        #7c5cff
    );
    color: white;
}

div.stButton > button:hover {
    border: none;
    transform: translateY(-1px);
}

[data-testid="stTextInput"] input {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown("""
<div class="hero-title">
📄 Doküman AI Asistanı
</div>

<div class="hero-subtitle">
PDF veya TXT dokümanınızı yükleyin.
Yapay zekâ dokümanı analiz etsin ve sorularınızı
yalnızca dokümandaki bilgilere göre yanıtlasın.
</div>

<div class="steps">
    <div class="step">① Dokümanı Yükle</div>
    <div class="step">② Sorunu Yaz</div>
    <div class="step">③ Yapay Zekâdan Cevap Al</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DOSYA YÜKLEME
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📎 Doküman Yükle</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "PDF veya TXT",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    st.success(
        f"✅ {uploaded_file.name} başarıyla yüklendi."
    )

    suffix = Path(uploaded_file.name).suffix
    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name


        with st.spinner("Doküman analiz ediliyor..."):

            text = read_document(temp_path)

            cleaned_text = clean_text(text)

            if not cleaned_text:

                st.error(
                    "Bu dokümandan okunabilir metin çıkarılamadı."
                )

                st.stop()


            chunks = chunk_text(
                cleaned_text,
                chunk_size=100,
                overlap=20
            )

            chunk_embeddings = create_embeddings(chunks)


        st.info(
            f"📚 Doküman hazır • "
            f"{len(chunks)} metin parçası analiz edildi."
        )


        # ------------------------------------------
        # SORU
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">💬 Doküman Hakkında Soru Sor</div>',
            unsafe_allow_html=True
        )

        question = st.text_input(
            "Sorunuz",
            placeholder="Örnek: Şirket nerede bulunuyor?",
            label_visibility="collapsed"
        )


        if st.button("✨ Cevabı Oluştur"):

            if not question.strip():

                st.warning(
                    "Lütfen önce bir soru yazın."
                )

            else:

                with st.spinner(
                    "Yapay zekâ dokümanda cevabı arıyor..."
                ):

                    results = find_relevant_chunks(
                        question,
                        chunks,
                        chunk_embeddings,
                        top_k=3
                    )


                    api_key = os.getenv(
                        "OPENAI_API_KEY"
                    )


                    if not api_key:

                        try:

                            api_key = st.secrets[
                                "OPENAI_API_KEY"
                            ]

                        except Exception:

                            api_key = None


                    if not api_key:

                        st.error(
                            "OpenAI API anahtarı bulunamadı."
                        )

                    else:

                        try:

                            answer = generate_answer(
                                question,
                                results,
                                api_key=api_key
                            )


                            # ----------------------
                            # CEVAP KARTI
                            # ----------------------

                            st.markdown(
                                f"""
                                <div class="answer-card">

                                    <div class="answer-label">
                                    🤖 YAPAY ZEKÂ CEVABI
                                    </div>

                                    <div class="answer-text">
                                    {answer}
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            st.caption(
                                f"📌 Kaynak: {uploaded_file.name}"
                            )


                            # ----------------------
                            # KAYNAKLAR
                            # ----------------------

                            with st.expander(
                                "🔎 Kullanılan doküman bölümlerini göster"
                            ):

                                for i, result in enumerate(
                                    results,
                                    start=1
                                ):

                                    st.markdown(
                                        f"""
                                        **Kaynak {i}**

                                        Benzerlik skoru:
                                        `{result['score']:.3f}`
                                        """
                                    )

                                    st.write(
                                        result["chunk"]
                                    )

                                    if i != len(results):
                                        st.divider()


                        except Exception as error:

                            st.error(
                                "Cevap oluşturulurken bir hata oluştu."
                            )

                            st.code(
                                str(error)
                            )


    finally:

        if temp_path and os.path.exists(
            temp_path
        ):

            os.remove(temp_path)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
Doküman Tabanlı Yapay Zekâ Asistanı
<br>
PDF • TXT • Semantic Search • Yapay Zekâ
</div>
""", unsafe_allow_html=True)