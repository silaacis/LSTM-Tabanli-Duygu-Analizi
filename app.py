import streamlit as st
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 200

model = load_model("sentiment_model.keras")
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = []

    for word in words:
        index = word_index.get(word)

        if index is not None and index < vocab_size:
            encoded.append(index + 3)
        else:
            encoded.append(2)

    return pad_sequences([encoded], maxlen=max_length)

def predict_sentiment(text):
    encoded_text = encode_review(text)
    prediction = model.predict(encoded_text)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, prediction

st.title("LSTM Tabanlı Duygu Analizi")
st.write("Bu uygulama, IMDB film yorumlarını pozitif veya negatif olarak sınıflandırır.")

review = st.text_area("Film yorumunu İngilizce olarak giriniz:")

if st.button("Analiz Et"):
    if review.strip() == "":
        st.warning("Lütfen bir yorum giriniz.")
    else:
        sentiment, score = predict_sentiment(review)

        st.subheader("Analiz Sonucu")

        if sentiment == "Positive":
            st.success(f"Pozitif yorum 😊")
        else:
            st.error(f"Negatif yorum 😟")

        st.write(f"Tahmin skoru: {score:.4f}")