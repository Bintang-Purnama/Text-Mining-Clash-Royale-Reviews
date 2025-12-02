# Text-Mining-Clash-Royale-Reviews
## Scrape and Analysis the results of the scrapping 
## ⚔️ Text Mining & Sentiment Analysis: Clash Royale Reviews

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Library](https://img.shields.io/badge/Library-Sklearn%20|%20NLTK%20|%20Gensim-orange.svg)

## 📝 Deskripsi Project

Project ini bertujuan untuk melakukan **Text Mining** dan **Analisis Sentimen** terhadap ulasan pengguna game *Clash Royale* di Google Play Store. Project ini mencakup pipeline *end-to-end* mulai dari pengambilan data (scraping), pembersihan teks (preprocessing), ekstraksi fitur, penanganan data tidak seimbang (imbalanced data), hingga pemodelan *Machine Learning* untuk memprediksi rating berdasarkan isi ulasan.

Masalah utama yang diselesaikan dalam project ini adalah klasifikasi sentimen pada dataset yang sangat tidak seimbang (didominasi rating 5), dengan membandingkan performa representasi teks **TF-IDF** vs **Word2Vec**.

## ✨ Fitur & Alur Kerja

Project ini terdiri dari 7 tahapan utama:

1.  **Data Scraping**: Mengambil 1000 ulasan terbaru dari Google Play Store menggunakan `google-play-scraper`.
2.  **Exploratory Data Analysis (EDA)**:
    * Visualisasi distribusi rating (terdeteksi *imbalance*).
    * Visualisasi kata yang sering muncul menggunakan **WordCloud**.
3.  **Text Preprocessing**:
    * *Case folding*, pembersihan angka/tanda baca.
    * *Stopword removal* (menghapus kata umum yang tidak bermakna).
    * *POS Tagging* & *Lemmatization* (mengubah kata ke bentuk dasar berdasarkan konteks tata bahasa).
4.  **Feature Extraction (Text Representation)**:
    * **TF-IDF**: Representasi berbasis frekuensi kata.
    * **Word2Vec**: Representasi berbasis vektor makna semantik (menggunakan model `gensim` yang dilatih sendiri).
5.  **Model Training (Baseline)**:
    * Algoritma: **Logistic Regression** dan **Random Forest**.
    * Tuning: Menggunakan `GridSearchCV` untuk mencari hyperparameter terbaik.
6.  **Handling Imbalanced Data**:
    * Menggunakan **SMOTE** (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan kelas minoritas (rating 1, 2, 3, 4) agar setara dengan kelas mayoritas.
7.  **Evaluasi Akhir**: Membandingkan performa model sebelum dan sesudah SMOTE menggunakan metrik *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.

## 🛠️ Teknologi yang Digunakan

* **Bahasa**: Python
* **Data Acquisition**: `google-play-scraper`
* **Data Processing**: `pandas`, `numpy`
* **NLP & Text Processing**: `nltk`, `re` (RegEx)
* **Feature Extraction**: `sklearn.feature_extraction.text` (TF-IDF), `gensim` (Word2Vec)
* **Machine Learning**: `scikit-learn` (Logistic Regression, Random Forest, SMOTE)
* **Visualization**: `matplotlib`, `wordcloud`

## 📂 Struktur File

File-file penting dalam repository ini:

* `Text_Mining_code.ipynb`: Notebook utama yang berisi seluruh kode dari scraping hingga evaluasi.
* `clashroyale.csv`: Data mentah hasil scraping (1000 ulasan).
* `clashroyale_cleaned.csv`: Data hasil preprocessing (teks bersih dan ter-lemmatisasi).
* `X_train_tfidf.npy`, `X_test_tfidf.npy`: Fitur hasil ekstraksi TF-IDF (disimpan dalam format numpy).
* `X_train_w2v.npy`, `X_test_w2v.npy`: Fitur hasil ekstraksi Word2Vec.
* `y_train.csv`, `y_test.csv`: Label target (rating) untuk data latih dan uji.

## 🚀 Cara Menjalankan

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/username/repo-name.git](https://github.com/username/repo-name.git)
    ```

2.  **Install library yang dibutuhkan:**
    Pastikan Anda memiliki Python terinstall, lalu jalankan:
    ```bash
    pip install pandas numpy matplotlib scikit-learn nltk gensim google-play-scraper wordcloud imbalanced-learn
    ```

3.  **Jalankan Notebook:**
    Buka `Text_Mining_code.ipynb` menggunakan Jupyter Notebook atau Google Colab.
    * **Note**: Anda bisa melewati tahap scraping (Langkah 1) jika ingin langsung menggunakan file `clashroyale.csv` yang sudah tersedia.

## 📊 Hasil Analisis

Berdasarkan eksperimen yang dilakukan:

1.  **Isu Data Imbalance**: Dataset awal didominasi oleh Rating 5. Hal ini menyebabkan Akurasi tinggi namun F1-Score rendah pada kelas minoritas.
2.  **Representasi Teks**:
    * **TF-IDF** terbukti lebih unggul dibandingkan Word2Vec rata-rata untuk dataset ini. TF-IDF mampu menangkap kata kunci spesifik (seperti "lag", "crash", "good") dengan lebih baik.
3.  **Performa Model**:
    * **Logistic Regression** dengan TF-IDF memberikan hasil yang paling stabil dan akurat.
    * Penerapan **SMOTE** berhasil menyeimbangkan distribusi kelas, yang bertujuan meningkatkan kemampuan model dalam mendeteksi ulasan negatif (Rating 1 & 2).

## 🤝 Kontribusi

Project ini dibuat oleh **Bintang Purnama**. Kritik dan saran sangat terbuka untuk pengembangan lebih lanjut.

## 📄 Lisensi

Project ini bersifat open-source di bawah lisensi [MIT](LICENSE).
