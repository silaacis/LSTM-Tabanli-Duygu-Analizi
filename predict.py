from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Parametreler
vocab_size = 10000
max_length = 200

# Kaydedilmiş modeli yükle
model = load_model("sentiment_model.keras")

# IMDB kelime indeksini al
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()

    encoded = []
    for word in words:
        index = word_index.get(word)

        if index is not None and index < vocab_size:
            encoded.append(index + 3)
        else:
            encoded.append(2)  # bilinmeyen kelime

    padded = pad_sequences([encoded], maxlen=max_length)
    return padded

def predict_sentiment(text):
    encoded_text = encode_review(text)
    prediction = model.predict(encoded_text)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, prediction

while True:
    review = input("\nEnter a movie review (or type 'q' to quit): ")

    if review.lower() == "q":
        print("Program closed.")
        break

    sentiment, score = predict_sentiment(review)

    print("Prediction:", sentiment)
    print(f"Score: {score:.4f}")