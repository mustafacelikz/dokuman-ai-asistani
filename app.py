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


# =========================================================
# SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Doküman AI Asistanı",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS TASARIM
# =========================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(78, 70, 229, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(37, 99, 235, 0.10),
            transparent 30%
        ),
        #0b0f17;
}

.block-container {
    max-width: 940px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* HERO */

.hero {
    padding-top: 10px;
    margin-bottom: 36px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    margin-bottom: 18px;

    border-radius: 999px;

    border: 1px solid rgba(129, 140, 248, 0.28);

    background: rgba(99, 102, 241, 0.09);

    color: #c7d2fe;

    font-size: 13px;
    font-weight: 600;
}

.hero-title {
    font-size: 58px;
    font-weight: 850;

    letter-spacing: -2px;
    line-height: 1.04;

    margin-bottom: 18px;

    background:
        linear-gradient(
            90deg,
            #ffffff 0%,
            #dbeafe 50%,
            #93c5fd 100%
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    max-width: 780px;

    color: #a5afbe;

    font-size: 17px;
    line-height: 1.7;
}

.steps {
    display: flex;
    flex-wrap: wrap;

    gap: 10px;

    margin-top: 28px;
}

.step {
    padding: 10px 16px;

    border-radius: 999px;

    border: 1px solid rgba(255,255,255,0.08);

    background: rgba(255,255,255,0.035);

    color: #d6dce5;

    font-size: 13px;
}

.step-number {
    color: #a5b4fc;
    font-weight: 800;

    margin-right: 6px;
}


/* SECTION */

.section-title {
    margin-top: 14px;
    margin-bottom: 5px;

    font-size: 21px;
    font-weight: 750;

    color: #f3f4f6;
}

.section-description {
    margin-bottom: 14px;

    color: #7f899a;

    font-size: 14px;
}


/* FILE UPLOADER */

[data-testid="stFileUploaderDropzone"] {
    padding: 24px;

    border-radius: 17px;

    border: 1px dashed rgba(129,140,248,0.55);

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.06),
            rgba(59,130,246,0.025)
        );
}


/* TEXT INPUT */

[data-testid="stTextInput"] input {
    height: 52px;

    border-radius: 13px;

    border: 1px solid rgba(255,255,255,0.10);

    background: rgba(255,255,255,0.04);
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(129,140,248,0.75);

    box-shadow:
        0 0 0 1px rgba(129,140,248,0.20);
}


/* BUTTON */

div.stButton > button {
    width: 100%;

    min-height: 50px;

    border-radius: 13px;

    border: none;

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
    border: none;

    transform: translateY(-1px);

    box-shadow:
        0 10px 30px rgba(79,70,229,0.22);
}


/* DOCUMENT INFO */

.document-ok {
    margin-top: 15px;

    padding: 16px 18px;

    border-radius: 14px;

    border: 1px solid rgba(34,197,94,0.19);

    background: rgba(34,197,94,0.075);

    color: #c8f7d5;

    font-size: 14px;
}

.stats {
    display: flex;
    flex-wrap: wrap;

    gap: 9px;

    margin-top: 13px;
    margin-bottom: 30px;
}

.stat {
    padding: 9px 13px;

    border-radius: 10px;

    border: 1px solid rgba(255,255,255,0.07);

    background: rgba(255,255,255,0.035);

    color: #9da8b8;

    font-size: 12px;
}


/* ANSWER */

.answer-card {
    margin-top: 25px;
    margin-bottom: 18px;

    padding: 27px;

    border-radius: 20px;

    border: 1px solid rgba(129,140,248,0.30);

    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.14),
            rgba(59,130,246,0.07)
        );

    box-shadow:
        0 20px 60px rgba(0,0,0,0.18);
}

.answer-label {
    margin-bottom: 12px;

    color: #a5b4fc;

    font-size: 13px;
    font-weight: 800;

    letter-spacing: .7px;
}

.answer-text {
    color: #f5f7fb;

    font-size: 20px;
    line-height: 1.65;
}

.answer-source {
    margin-top: 18px;

    color: #8490a2;

    font-size: 13px;
}


/* FOOTER */

.footer {
    margin-top: 65px;

    padding-top: 24px;

    border-top: 1px solid rgba(255,255,255,0.06);

    text-align: center;

    color: #596476;

    font-size: 12px;
    line-height: 1.8;
}


/* STREAMLIT HEADER */

header[data-testid="stHeader"] {
    background: transparent;
}

</style>
""")


# =========================================================
# ÜST TASARIM
# =========================================================

st.html("""
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
""")


# =========================================================
# DOKÜMAN YÜKLEME
# =========================================================

st.html("""
<div class="section-title">
    📎 Doküman Yükle
</div>

<div class="section-description">
    Analiz etmek istediğiniz PDF veya TXT dosyasını seçin.
</div>
""")


uploaded_file = st.file_uploader(
    "Doküman seçin",
    type=["pdf", "txt"],
    label_visibility="collapsed"
)


# =========================================================
# DOSYA YÜKLENDİYSE
# =========================================================

if uploaded_file is not None:

    safe_filename = html.escape(
        uploaded_file.name
    )

    st.html(
        f"""
        <div class="document-ok">
            ✅ <b>{safe_filename}</b> başarıyla yüklendi.
        </div>
        """
    )

    suffix = Path(
        uploaded_file.name
    ).suffix

    temp_path = None

    try:

        # -------------------------------------------------
        # GEÇİCİ DOSYA
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name


        # -------------------------------------------------
        # DOKÜMAN ANALİZİ
        # -------------------------------------------------

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


        # -------------------------------------------------
        # İSTATİSTİKLER
        # -------------------------------------------------

        word_count = len(
            cleaned_text.split()
        )


        st.html(
            f"""
            <div class="stats">

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
            """
        )


        # =================================================
        # SORU ALANI
        # =================================================

        st.html("""
        <div class="section-title">
            💬 Doküman Hakkında Soru Sor
        </div>

        <div class="section-description">
            Sorunuzun cevabı yüklediğiniz dokümanda aranacaktır.
        </div>
        """)


        question = st.text_input(
            "Sorunuz",
            placeholder="Örnek: Şirket nerede bulunuyor?",
            label_visibility="collapsed"
        )


        # =================================================
        # CEVAP
        # =================================================

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

                    # -------------------------------------
                    # RETRIEVAL
                    # -------------------------------------

                    results = find_relevant_chunks(
                        question,
                        chunks,
                        chunk_embeddings,
                        top_k=3
                    )


                    # -------------------------------------
                    # API KEY
                    # -------------------------------------

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


                    # -------------------------------------
                    # API ANAHTARI YOK
                    # -------------------------------------

                    if not api_key:

                        st.error(
                            "OpenAI API anahtarı bulunamadı."
                        )


                    # -------------------------------------
                    # CEVAP ÜRET
                    # -------------------------------------

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


                            st.html(
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
                                """
                            )


                            # -----------------------------
                            # KAYNAKLAR
                            # -----------------------------

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

**Benzerlik skoru:** `{result['score']:.3f}`
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


    # =====================================================
    # GEÇİCİ DOSYAYI TEMİZLE
    # =====================================================

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    <b>Doküman AI Asistanı</b>
    <br>

    PDF / TXT • Semantic Search • Embedding • Yapay Zekâ

</div>
""")