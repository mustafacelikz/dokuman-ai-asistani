import os
import html
import tempfile
from pathlib import Path

import streamlit as st

from src.document_reader import read_document
from src.text_processor import clean_text, chunk_text
from src.embedding_manager import create_embeddings
from src.retriever import find_relevant_chunks
from src.answer_generator import generate_answer


# ==================================================
# SAYFA AYARLARI
# ==================================================

st.set_page_config(
    page_title="Doküman AI Asistanı",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==================================================
# TASARIM / CSS
# ==================================================

st.markdown(
    """
    <style>

    /* ANA SAYFA */
    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(79, 70, 229, 0.15),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 5%,
                rgba(59, 130, 246, 0.10),
                transparent 25%
            ),
            #0b0f17;
    }

    .block-container {
        max-width: 940px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }


    /* HERO */
    .hero {
        padding: 10px 0 10px 0;
        margin-bottom: 8px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 8px 14px;

        border-radius: 999px;

        background: rgba(99, 102, 241, 0.10);

        border: 1px solid rgba(129, 140, 248, 0.22);

        color: #c7d2fe;

        font-size: 13px;
        font-weight: 600;

        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 54px;
        font-weight: 850;
        letter-spacing: -1.8px;
        line-height: 1.05;

        margin: 0;

        background:
            linear-gradient(
                90deg,
                #ffffff 0%,
                #e0e7ff 45%,
                #93c5fd 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        margin-top: 18px;

        max-width: 760px;

        font-size: 17px;
        line-height: 1.7;

        color: #9ca9bd;
    }


    /* ADIMLAR */
    .steps {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;

        margin-top: 28px;
        margin-bottom: 38px;
    }

    .step {
        padding: 10px 15px;

        border-radius: 999px;

        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.08);

        color: #d2d8e2;

        font-size: 13px;
        font-weight: 500;
    }

    .step-number {
        color: #a5b4fc;
        font-weight: 800;
        margin-right: 5px;
    }


    /* BÖLÜM BAŞLIKLARI */
    .section-header {
        display: flex;
        align-items: center;

        gap: 10px;

        margin-top: 18px;
        margin-bottom: 12px;

        font-size: 20px;
        font-weight: 750;

        color: #f3f4f6;
    }

    .section-description {
        color: #7f8a9c;
        font-size: 14px;
        margin-bottom: 14px;
    }


    /* UPLOADER */
    [data-testid="stFileUploader"] {
        margin-bottom: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        padding: 22px;

        border-radius: 17px;

        border: 1px dashed rgba(129, 140, 248, 0.55);

        background:
            linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.055),
                rgba(59, 130, 246, 0.025)
            );
    }


    /* INPUT */
    [data-testid="stTextInput"] input {
        height: 52px;

        border-radius: 13px;

        border: 1px solid rgba(255,255,255,0.10);

        background: rgba(255,255,255,0.035);

        font-size: 15px;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: rgba(129, 140, 248, 0.70);
        box-shadow: 0 0 0 1px rgba(129, 140, 248, 0.25);
    }


    /* BUTTON */
    div.stButton > button {
        width: 100%;

        min-height: 50px;

        border-radius: 13px;

        border: 1px solid rgba(255,255,255,0.06);

        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #6366f1,
                #3b82f6
            );

        color: white;

        font-size: 15px;
        font-weight: 700;

        transition: 0.18s ease;
    }

    div.stButton > button:hover {
        border: 1px solid rgba(255,255,255,0.14);

        transform: translateY(-1px);

        box-shadow:
            0 8px 30px rgba(79, 70, 229, 0.20);
    }


    /* DURUM KARTI */
    .document-card {
        margin-top: 14px;

        padding: 16px 18px;

        border-radius: 14px;

        background: rgba(34, 197, 94, 0.075);

        border: 1px solid rgba(34, 197, 94, 0.18);

        color: #c7f9d4;

        font-size: 14px;
    }


    /* AI CEVAP KARTI */
    .answer-card {
        position: relative;

        margin-top: 26px;
        margin-bottom: 16px;

        padding: 26px;

        border-radius: 20px;

        border: 1px solid rgba(129, 140, 248, 0.28);

        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.13),
                rgba(59,130,246,0.065)
            );

        box-shadow:
            0 20px 60px rgba(0,0,0,0.18);
    }

    .answer-label {
        display: flex;
        align-items: center;
        gap: 8px;

        margin-bottom: 13px;

        color: #a5b4fc;

        font-size: 13px;
        font-weight: 800;

        letter-spacing: 0.5px;
    }

    .answer-text {
        font-size: 20px;
        line-height: 1.65;

        color: #f4f7fb;
    }

    .answer-source {
        margin-top: 18px;

        color: #8490a3;

        font-size: 13px;
    }


    /* İSTATİSTİKLER */
    .stat-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;

        margin-top: 13px;
        margin-bottom: 25px;
    }

    .stat {
        padding: 9px 13px;

        border-radius: 10px;

        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.07);

        color: #9da9bc;

        font-size: 12px;
    }


    /* FOOTER */
    .footer {
        margin-top: 60px;

        padding-top: 24px;

        border-top: 1px solid rgba(255,255,255,0.06);

        text-align: center;

        color: #566174;

        font-size: 12px;
        line-height: 1.8;
    }


    /* STREAMLIT ÜST BOŞLUĞU */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# ÜST ALAN
# ==================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✨ Yapay Zekâ Destekli Doküman Analizi
        </div>

        <div class="hero-title">
            Doküman AI Asistanı
        </div>

        <div class="hero-subtitle">
            PDF veya TXT dokümanınızı yükleyin.
            Sistem dokümanı analiz etsin, ilgili bölümleri bulsun
            ve sorularınızı yalnızca yüklediğiniz dokümandaki
            bilgilere göre yanıtlasın.
        </div>

        <div class="steps">

            <div class="step">
                <span class="step-number">01</span>
                Dokümanı Yükle
            </div>

            <div class="step">
                <span class="step-number">02</span>
                Sorunu Yaz
            </div>

            <div class="step">
                <span class="step-number">03</span>
                AI Cevabını Al
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# DOSYA YÜKLEME
# ==================================================

st.markdown(
    """
    <div class="section-header">
        📎 Doküman Yükle
    </div>

    <div class="section-description">
        Analiz etmek istediğiniz PDF veya TXT dosyasını seçin.
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Doküman seçin",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)


# ==================================================
# DOSYA VARSA İŞLE
# ==================================================

if uploaded_file is not None:

    safe_filename = html.escape(uploaded_file.name)

    st.markdown(
        f"""
        <div class="document-card">
            ✅ <b>{safe_filename}</b> başarıyla yüklendi.
        </div>
        """,
        unsafe_allow_html=True
    )

    suffix = Path(uploaded_file.name).suffix

    temp_path = None

    try:

        # ------------------------------------------
        # GEÇİCİ DOSYA
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name


        # ------------------------------------------
        # DOKÜMANI HAZIRLA
        # ------------------------------------------

        with st.spinner(
            "Doküman analiz ediliyor..."
        ):

            text = read_document(
                temp_path
            )

            cleaned_text = clean_text(
                text
            )


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


            chunk_embeddings = create_embeddings(
                chunks
            )


        # ------------------------------------------
        # DOKÜMAN BİLGİSİ
        # ------------------------------------------

        word_count = len(
            cleaned_text.split()
        )

        st.markdown(
            f"""
            <div class="stat-container">

                <div class="stat">
                    📚 {len(chunks)} metin parçası
                </div>

                <div class="stat">
                    📝 {word_count} kelime
                </div>

                <div class="stat">
                    ✅ Analiz tamamlandı
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ==================================================
        # SORU ALANI
        # ==================================================

        st.markdown(
            """
            <div class="section-header">
                💬 Doküman Hakkında Soru Sor
            </div>

            <div class="section-description">
                Sorunuzun cevabı yüklediğiniz dokümanda aranacaktır.
            </div>
            """,
            unsafe_allow_html=True
        )


        question = st.text_input(
            "Sorunuz",
            placeholder="Örnek: Şirket nerede bulunuyor?",
            label_visibility="collapsed"
        )


        # ==================================================
        # CEVAP BUTONU
        # ==================================================

        if st.button(
            "✨ Yapay Zekâ Cevabını Oluştur"
        ):

            if not question.strip():

                st.warning(
                    "Lütfen önce bir soru yazın."
                )

            else:

                with st.spinner(
                    "Yapay zekâ dokümanda ilgili bilgileri arıyor..."
                ):

                    # --------------------------------------
                    # İLGİLİ CHUNKLARI BUL
                    # --------------------------------------

                    results = find_relevant_chunks(
                        question,
                        chunks,
                        chunk_embeddings,
                        top_k=3
                    )


                    # --------------------------------------
                    # API ANAHTARI
                    # --------------------------------------

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


                    # --------------------------------------
                    # ANAHTAR YOK
                    # --------------------------------------

                    if not api_key:

                        st.error(
                            "OpenAI API anahtarı bulunamadı."
                        )


                    # --------------------------------------
                    # CEVAP ÜRET
                    # --------------------------------------

                    else:

                        try:

                            answer = generate_answer(
                                question,
                                results,
                                api_key=api_key
                            )


                            safe_answer = html.escape(
                                answer
                            )


                            # ----------------------------------
                            # AI CEVAP KARTI
                            # ----------------------------------

                            st.markdown(
                                f"""
                                <div class="answer-card">

                                    <div class="answer-label">
                                        🤖 AI ASİSTAN CEVABI
                                    </div>

                                    <div class="answer-text">
                                        {safe_answer}
                                    </div>

                                    <div class="answer-source">
                                        📌 Kaynak doküman:
                                        {safe_filename}
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                            # ----------------------------------
                            # KAYNAKLAR
                            # ----------------------------------

                            with st.expander(
                                "🔎 Kullanılan doküman bölümlerini incele"
                            ):

                                for i, result in enumerate(
                                    results,
                                    start=1
                                ):

                                    st.markdown(
                                        f"""
                                        ### Kaynak {i}

                                        **Benzerlik skoru:**
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
                                "Cevap oluşturulurken bir hata meydana geldi."
                            )

                            st.code(
                                str(error)
                            )


    # ==================================================
    # GEÇİCİ DOSYAYI SİL
    # ==================================================

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">

        <b>Doküman AI Asistanı</b>
        <br>

        PDF / TXT • Semantic Search • Embedding • Yapay Zekâ

    </div>
    """,
    unsafe_allow_html=True
)