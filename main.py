import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parametreler
vocab_size = 30000
max_length = 100
embedding_dim = 128

# Veri setini oku
df = pd.read_csv("turkish_ecommerce_reviews.csv")

# Sadece gerekli sütunları al
df = df[["Review", "Rating (Star)"]]

# Boş yorumları temizle
df = df.dropna()

# 3 yıldızlı yorumları çıkarıyoruz çünkü nötr/kararsız kabul edilebilir
df = df[df["Rating (Star)"] != 3]

# Etiket oluşturma
# 1-2 yıldız: Negatif -> 0
# 4-5 yıldız: Pozitif -> 1
df["label"] = df["Rating (Star)"].apply(lambda x: 1 if x >= 4 else 0)

# Metin temizleme fonksiyonu
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["Review"].apply(clean_text)

# Çok kısa yorumları çıkar
df = df[df["clean_review"].str.len() > 3]

print("Veri boyutu:", df.shape)
print("Etiket dağılımı:")
print(df["label"].value_counts())

# Veri çok dengesiz olduğu için pozitif ve negatif sınıfları dengeliyoruz
negative_df = df[df["label"] == 0]
positive_df = df[df["label"] == 1].sample(len(negative_df), random_state=42)

df = pd.concat([negative_df, positive_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDengelenmiş veri boyutu:", df.shape)
print("Dengelenmiş etiket dağılımı:")
print(df["label"].value_counts())

# X ve y
X = df["clean_review"].astype(str).to_numpy()
y = df["label"].astype(int).to_numpy()

# Eğitim-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tokenizer oluşturma
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Metinleri sayısal dizilere çevirme
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding="post", truncating="post")

# Tokenizer kaydetme
with open("turkish_tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Model oluşturma
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Model derleme
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

# Model eğitimi
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Test değerlendirme
loss, accuracy = model.evaluate(X_test_pad, y_test)

# Tahminler
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negatif", "Pozitif"]))

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Model kaydetme
model.save("turkish_sentiment_model.keras")
print("Türkçe duygu analizi modeli başarıyla kaydedildi.")

# Accuracy grafiği
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Türkçe BiLSTM Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss grafiği
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Türkçe BiLSTM Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Negatif", "Pozitif"])
plt.yticks([0, 1], ["Negatif", "Pozitif"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()