# 🛒 Türkçe Ürün Yorumu Duygu Analizi (BiLSTM)

Bu proje, Türkçe e-ticaret ürün yorumlarının duygu analizini gerçekleştirmek amacıyla geliştirilmiştir. Projede derin öğrenme tabanlı **Bidirectional Long Short-Term Memory (BiLSTM)** mimarisi kullanılarak kullanıcı yorumlarının pozitif veya negatif olarak sınıflandırılması amaçlanmıştır.

## 📌 Proje Amacı

E-ticaret platformlarında kullanıcılar tarafından yapılan yorumlar, ürün kalitesi ve müşteri memnuniyeti hakkında önemli bilgiler içermektedir. Bu proje, ürün yorumlarının otomatik olarak analiz edilerek duygu durumlarının belirlenmesini amaçlamaktadır.

## 🚀 Kullanılan Teknolojiler

* Python
* TensorFlow / Keras
* BiLSTM (Bidirectional LSTM)
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Streamlit

## 📊 Veri Seti

Projede Türkçe e-ticaret ürün yorumlarından oluşan bir veri seti kullanılmıştır.

Veri ön işleme aşamasında:

* Eksik veriler temizlenmiştir.
* Metinler küçük harfe dönüştürülmüştür.
* Noktalama işaretleri ve özel karakterler kaldırılmıştır.
* 1-2 yıldızlı yorumlar **Negatif**, 4-5 yıldızlı yorumlar **Pozitif** olarak etiketlenmiştir.
* 3 yıldızlı yorumlar nötr kabul edilerek veri setinden çıkarılmıştır.
* Sınıf dengesizliğini önlemek amacıyla veri seti dengelenmiştir.

## 🧠 Model Mimarisi

Model aşağıdaki katmanlardan oluşmaktadır:

```text
Embedding Layer
        ↓
Bidirectional LSTM
        ↓
Dropout
        ↓
Dense (Sigmoid)
```

### Model Parametreleri

* Vocabulary Size: 30.000
* Embedding Dimension: 128
* LSTM Units: 64
* Dropout Rate: 0.5
* Optimizer: Adam
* Loss Function: Binary Crossentropy

## 📈 Model Performansı

Test veri seti üzerinde elde edilen sonuçlar:

| Metrik    | Değer  |
| --------- | ------ |
| Accuracy  | %88.81 |
| F1-Score  | %89    |
| Precision | %89    |
| Recall    | %89    |

### Classification Report

```text
Negatif
Precision: 0.87
Recall:    0.92
F1-Score:  0.89

Pozitif
Precision: 0.91
Recall:    0.86
F1-Score:  0.88
```

## 📉 Confusion Matrix

```text
                Tahmin
              Neg    Poz
Gerçek Neg   2504   231
Gerçek Poz    381  2353
```

Model her iki sınıfta da dengeli performans göstermiştir.

## 🖥️ Uygulama Arayüzü

Projede Streamlit kullanılarak kullanıcı dostu bir web arayüzü geliştirilmiştir.

Kullanıcı:

1. Ürün yorumunu girer.
2. "Analiz Et" butonuna basar.
3. Model yorumun duygu durumunu tahmin eder.
4. Sonuç ve güven seviyesi ekranda gösterilir.

## ⚙️ Kurulum

Projeyi çalıştırmak için:

```bash
pip install -r requirements.txt
```

Ardından:

```bash
streamlit run app.py
```

komutunu çalıştırın.

## 📂 Proje Yapısı

```text
├── app.py
├── main.py
├── requirements.txt
├── turkish_sentiment_model.keras
├── turkish_tokenizer.pkl
├── README.md
└── .gitignore
```

## 🎯 Sonuç

Bu projede Türkçe ürün yorumlarının duygu analizi için BiLSTM tabanlı bir derin öğrenme modeli geliştirilmiştir. Model, test verileri üzerinde %88.81 doğruluk oranına ulaşmış ve pozitif/negatif yorumları başarılı şekilde sınıflandırmıştır. Ayrıca geliştirilen Streamlit arayüzü sayesinde model gerçek zamanlı olarak kullanılabilmektedir.

## 👩‍💻 Geliştirici

**Sıla Açış**
Bilgisayar Mühendisliği Öğrencisi
Bursa Uludağ Üniversitesi
