import re
import pickle
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ayarlar
vocab_size = 30000
max_length = 100

# Model yükleme
model = load_model("turkish_sentiment_model.keras")

# Tokenizer yükleme
with open("turkish_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Metin temizleme
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tahmin fonksiyonu
def predict_sentiment(text):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    score = model.predict(padded, verbose=0)[0][0]

    if score < 0.20:
        sentiment = "Çok Negatif 😠"
        confidence = "Çok Yüksek"
    elif score < 0.40:
        sentiment = "Negatif 😟"
        confidence = "Yüksek"
    elif score < 0.60:
        sentiment = "Kararsız / Nötre Yakın ⚠️"
        confidence = "Düşük"
    elif score < 0.80:
        sentiment = "Pozitif 🙂"
        confidence = "Yüksek"
    else:
        sentiment = "Çok Pozitif 😊"
        confidence = "Çok Yüksek"

    return sentiment, score, confidence

# Streamlit Arayüzü
st.set_page_config(
    page_title="Türkçe Duygu Analizi",
    page_icon="🛒"
)

st.title("🛒 Türkçe Ürün Yorumu Duygu Analizi")

st.write(
    "Bu uygulama Türkçe ürün yorumlarını "
    "BiLSTM modeli kullanarak analiz eder."
)

review = st.text_area(
    "Ürün yorumunu giriniz:"
)

if st.button("Analiz Et"):

    if review.strip() == "":
        st.warning("Lütfen bir yorum giriniz.")
    else:

        sentiment, score, confidence = predict_sentiment(review)

        st.subheader("📊 Analiz Sonucu")

        if score < 0.20:
            st.error(sentiment)

        elif score < 0.40:
            st.error(sentiment)

        elif score < 0.60:
            st.warning(sentiment)

        elif score < 0.80:
            st.success(sentiment)

        else:
            st.success(sentiment)

        st.write(f"**Duygu:** {sentiment}")
        st.write(f"**Skor:** {score:.4f}")
        st.write(f"**Güven Seviyesi:** {confidence}")

        st.progress(float(score))

        st.info(
            """
            Skor Yorumu:

            • 0.00 - 0.20 → Çok Negatif  
            • 0.20 - 0.40 → Negatif  
            • 0.40 - 0.60 → Kararsız / Nötre Yakın  
            • 0.60 - 0.80 → Pozitif  
            • 0.80 - 1.00 → Çok Pozitif
            """
        )