# Laporan Proyek Machine Learning - Ghani Husna Daramwan

## Project Overview

Di era digital yang didominasi oleh konten hiburan, anime telah menjadi salah satu bentuk media yang paling populer secara global. Dengan ribuan judul yang tersedia di berbagai platform, pengguna seringkali menghadapi fenomena *overload informasi*, di mana menemukan anime baru yang sesuai dengan selera pribadi menjadi tantangan tersendiri. Fenomena ini diperparah dengan pertumbuhan eksponensial dalam produksi anime, menjadikannya semakin sulit bagi penggemar untuk menjelajahi dan menemukan judul-judul yang menarik.

Proyek ini bertujuan untuk mengatasi masalah ini dengan membangun sistem rekomendasi anime yang personal dan efektif. Sistem ini akan membantu pengguna menemukan anime yang relevan berdasarkan preferensi mereka, baik dari sisi karakteristik anime (genre, tipe) maupun dari perilaku tontonan pengguna lain. Sistem rekomendasi sangat vital dalam meningkatkan pengalaman pengguna pada platform streaming anime atau situs basis data anime, karena dapat secara signifikan meningkatkan *engagement* pengguna dan memperpanjang durasi mereka dalam menjelajahi konten. Penelitian menunjukkan bahwa sistem rekomendasi yang akurat dapat meningkatkan konsumsi konten dan kepuasan pengguna [1].

**Referensi**:  
[1] J. L. Herlocker, J. A. Konstan, A. Borchers, and J. Riedl, "An algorithmic framework for performing collaborative filtering," in *Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval*, 1999, pp. 230-237.

## Business Understanding

### Problem Statements
- **Overload Informasi**: Pengguna kesulitan menemukan anime baru yang relevan dan menarik di tengah jutaan pilihan yang tersedia di berbagai platform, yang dapat menyebabkan kelelahan keputusan dan mengurangi kepuasan pengguna.
- **Kurangnya Personalisasi**: Platform penyedia anime seringkali gagal menyediakan rekomendasi yang sangat personal, yang dapat menyebabkan pengguna merasa tidak mendapatkan nilai maksimal dari layanan dan mungkin beralih ke platform lain.

### Goals
- **Mempermudah Penemuan Konten**: Membangun sistem yang mampu merekomendasikan anime yang sangat relevan dan menarik kepada pengguna secara efisien, mengurangi waktu dan usaha yang dibutuhkan untuk menemukan konten baru.
- **Meningkatkan Kepuasan Pengguna**: Menyediakan rekomendasi yang personal dan akurat, sehingga meningkatkan pengalaman pengguna secara keseluruhan dan mendorong mereka untuk menjelajahi lebih banyak konten di platform.

### Solution Statements
Untuk mencapai tujuan di atas, proyek ini mengimplementasikan dua pendekatan sistem rekomendasi:
1. **Content-based Filtering**: Merekomendasikan anime berdasarkan kemiripan atribut intrinsik (genre, tipe) menggunakan *Cosine Similarity* pada fitur genre yang diubah menjadi representasi numerik dengan *TF-IDF*.
2. **Collaborative Filtering**: Merekomendasikan anime berdasarkan pola rating pengguna serupa menggunakan *Singular Value Decomposition (SVD)*, sebuah teknik *matrix factorization* untuk prediksi rating.

## Data Understanding
Dataset yang digunakan adalah **Anime Recommendations Database** dari Kaggle: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Dataset ini terdiri dari dua file CSV: `anime.csv` dan `rating.csv`.

**Informasi Dataset**:
- **anime.csv**:
  - Jumlah data: 12,294 entri.
  - Kolom:
    - `anime_id`: ID unik untuk setiap anime (int64).
    - `name`: Nama anime (object).
    - `genre`: Genre anime (object, 62 nilai null).
    - `type`: Tipe anime (object, 25 nilai null).
    - `episodes`: Jumlah episode (object).
    - `rating`: Rata-rata rating komunitas (float64, 230 nilai null).
    - `members`: Jumlah anggota komunitas yang menambahkan anime ke daftar mereka (int64).
- **rating.csv**:
  - Jumlah data: 7,813,737 entri.
  - Kolom:
    - `user_id`: ID unik pengguna (int64).
    - `anime_id`: ID anime yang diberi rating (int64).
    - `rating`: Rating pengguna (int64, -1 hingga 10, -1 menunjukkan "ditonton tapi tidak dinilai").

**Exploratory Data Analysis**:
- **Jumlah Genre Unik**: 43 genre.
- **Top 10 Most Frequent Anime Types**:
  - TV: 3,787
  - OVA: 3,311
  - Movie: 2,348
  - Special: 1,676
  - ONA: 659
  - Music: 488
- **Distribution of Ratings (excluding -1)**:
  - 1: 16,649
  - 2: 23,150
  - 3: 41,453
  - 4: 104,291
  - 5: 282,806
  - 6: 637,775
  - 7: 1,375,287
  - 8: 1,646,019
  - 9: 1,254,096
  - 10: 955,715
- **Visualisasi**:
  - ![Distribution of Anime Ratings (excluding -1)](attachment://Distribution_of_Anime_Ratings_excluding_-1.png)
  - ![Top 15 Most Frequent Anime Genres](attachment://Top_15_Most_Frequent_Anime_Genres.png)

## Data Preparation
Berikut adalah tahapan persiapan data yang dilakukan:
1. **Penanganan Missing Values pada anime_df**:
   - Kolom `genre` (62 null) diisi dengan 'Unknown'.
   - Kolom `rating` (230 null) diisi dengan rata-rata rating.
   - **Alasan**: Mencegah error atau bias pada model. 'Unknown' menjaga informasi kategori, dan rata-rata rating mempertahankan distribusi data.
2. **Penanganan Rating -1 pada rating_df**:
   - Baris dengan rating -1 dihapus (dari 7,813,737 menjadi 6,337,239 baris).
   - **Alasan**: Rating -1 tidak memberikan informasi preferensi untuk *collaborative filtering*.
3. **Penanganan Duplikat pada anime_df**:
   - Duplikat berdasarkan `anime_id` dihapus (0 duplikat ditemukan).
   - **Alasan**: Menghindari redundansi dan kesalahan perhitungan.
4. **Penggabungan Dataset**:
   - `rating_df` digabungkan dengan `anime_df` berdasarkan `anime_id`, dengan suffix `_user` dan `_anime` untuk kolom rating (hasil: 6,337,239 baris, 9 kolom).
   - **Alasan**: Menghubungkan rating pengguna dengan detail anime untuk sistem rekomendasi.
5. **Pembersihan Kolom Genre untuk TF-IDF**:
   - Kolom `genre` dibersihkan dengan mengganti koma menjadi spasi, disimpan di `genre_cleaned`.
   - **Alasan**: Memastikan *TfidfVectorizer* memproses genre sebagai token terpisah.

**Contoh Kode**:
```python
anime_df['genre'] = anime_df['genre'].fillna('Unknown')
anime_df['rating'] = anime_df['rating'].fillna(anime_df['rating'].mean())
rating_df = rating_df[rating_df['rating'] != -1]
merged_df = pd.merge(rating_df, anime_df, on='anime_id', suffixes=('_user', '_anime'))
anime_df['genre_cleaned'] = anime_df['genre'].apply(lambda x: ' '.join(x.replace(',', ' ').split()))
```

## Modeling
### Content-based Filtering
- **Algoritma**: *Cosine Similarity*.
- **Cara Kerja**:
  - *TF-IDF Vectorization*: Kolom `genre_cleaned` diubah menjadi matriks *TF-IDF* (12,294 anime, 43 fitur genre).
  - *Cosine Similarity*: Menghitung kemiripan kosinus antar anime (matriks 12,294 x 12,294).
  - **Rekomendasi**: Anime dengan skor kemiripan tertinggi direkomendasikan.
- **Contoh Rekomendasi (Death Note)**:
  ```plaintext
  name                                              genre     type  rating  similarity_score
  Death Note Rewrite  Mystery, Police, Psychological, Supernatural, ...  Special 7.84 1.000000
  Mousou Dairinin     Drama, Mystery, Police, Psychological, Superna...  TV      7.74 0.967703
  Higurashi no Naku Koro ni Kai Mystery, Psychological, Supernatural, Thriller  TV      8.41 0.879514
  ...
- **Kelebihan**:
  - Tidak memerlukan data rating pengguna lain (*cold-start* friendly).
  - Rekomendasi transparan dan mudah dijelaskan.
  - Merekomendasikan item yang belum di-rated.
- **Kekurangan**:
  - *Over-specialization*, kurang beragam.
  - Memerlukan fitur item yang kaya.

### Collaborative Filtering
- **Algoritma**: *Singular Value Decomposition (SVD)* dari library *Surprise*.
- **Cara Kerja**:
  - Data rating dimuat ke format *Surprise* (skala 1-10).
  - Model SVD dilatih untuk memfaktorkan matriks rating pengguna-item.
  - **Rekomendasi**: Prediksi rating untuk anime yang belum ditonton, urutkan berdasarkan prediksi tertinggi.
- **Contoh Rekomendasi (user_id: 58625)**:
  ```plaintext
  anime_id  name                                    genre  type  average_anime_rating  predicted_rating
  30276     One Punch Man                           Action, Comedy, ...  TV  8.79  9.020295
  16498     Shingeki no Kyojin                      Action, Drama, ...   TV  8.47  8.986608
  1535      Death Note                              Mystery, Police, ... TV  8.71  8.974955
  ...
  ```
- **Kelebihan**:
  - Menemukan item baru tanpa kemiripan fitur (*serendipity*).
  - Tidak memerlukan metadata item.
- **Kekurangan**:
  - *Cold-start* problem untuk pengguna/item baru.
  - Masalah *sparsity* pada matriks rating.
  - Sulit menjelaskan rekomendasi.

**Contoh Kode**:
```python
from surprise import SVD
algo = SVD(random_state=42)
algo.fit(trainset)
def get_collaborative_recommendations(user_id, num_recommendations=10, df_anime=anime_df, df_merged=merged_df, svd_model=algo):
    # ... (fungsi lengkap seperti di source code)
    return pd.DataFrame(final_recommendations)
```

## Evaluation
### Metrik Evaluasi
1. **RMSE (Root Mean Squared Error)**:
   - **Formula**: `RMSE = sqrt(1/N * Σ(yi - ŷi)²)`
   - **Cara Kerja**: Mengukur rata-rata magnitudo error antara rating aktual dan prediksi. Memberikan bobot lebih pada error besar.
2. **MAE (Mean Absolute Error)**:
   - **Formula**: `MAE = 1/N * Σ|yi - ŷi|`
   - **Cara Kerja**: Mengukur rata-rata nilai absolut error, lebih resisten terhadap *outlier*.

### Hasil Evaluasi
- **Collaborative Filtering (SVD)**:
  - 5-fold cross-validation:
    ```plaintext
    Metric      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
    RMSE        1.1336  1.1332  1.1337  1.1348  1.1317  1.1334  0.0010
    MAE         0.8459  0.8445  0.8458  0.8461  0.8441  0.8453  0.0008
    Fit time    61.60   65.40   61.55   59.31   59.25   61.42   2.24
    Test time   13.07   14.44   12.21   12.27   12.28   12.86   0.85
    ```
  - **Mean RMSE**: 1.1334 (prediksi menyimpang ~1.13 unit).
  - **Mean MAE**: 0.8453.
  - **Kesimpulan**: Model SVD akurat dan konsisten (standar deviasi rendah).
- **Content-based Filtering**:
  - Evaluasi kualitatif: Rekomendasi untuk *Death Note* relevan (genre *Mystery*, *Psychological*, *Thriller*).
  - **Kesimpulan**: Efektif mengidentifikasi kemiripan genre.

**Kesimpulan Umum**: SVD memberikan prediksi rating akurat, sementara *Content-based Filtering* menghasilkan rekomendasi tematik yang relevan. Kedua pendekatan saling melengkapi untuk sistem rekomendasi anime yang komprehensif.

**---Ini adalah bagian akhir laporan---**

_Catatan:_  
- Visualisasi seperti "Distribution of Anime Ratings (excluding -1)" dan "Top 15 Most Frequent Anime Genres" telah disertakan sebagai gambar.  
- Untuk menjalankan proyek, instal library: `pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn`, unduh dataset dari [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database), simpan di folder `dataset/`, dan jalankan kode di Jupyter Notebook.
```

**Penjelasan Tambahan:**
- Laporan telah diperluas dengan menyertakan semua output yang Anda berikan (shape data, info dataframe, distribusi rating, dll.) di bagian yang relevan.
- Visualisasi gambar telah disisipkan dengan sintaks Markdown `![nama](attachment://nama.png)` berdasarkan file yang Anda unggah. Pastikan gambar disimpan di direktori yang sama dengan file README atau sesuaikan path jika diperlukan.
- Kode sumber lengkap Anda telah diringkas menjadi snippet relevan di bagian Modeling dan Data Preparation untuk menjelaskan logika tanpa menampilkan keseluruhan kode.
- Semua informasi dari output Anda (misalnya, jumlah baris setelah preprocessing, hasil cross-validation) telah diintegrasikan untuk memberikan laporan yang lebih komprehensif.