```markdown
# Proyek Sistem Rekomendasi Anime

## Daftar Isi
- [Pendahuluan](#pendahuluan)
- [Pemahaman Bisnis](#pemahaman-bisnis)
  - [Pernyataan Masalah](#pernyataan-masalah)
  - [Tujuan Proyek](#tujuan-proyek)
  - [Pendekatan Solusi](#pendekatan-solusi)
- [Pemahaman Data](#pemahaman-data)
- [Persiapan Data](#persiapan-data)
- [Pemodelan](#pemodelan)
  - [Content-based Filtering](#content-based-filtering)
  - [Collaborative Filtering](#collaborative-filtering)
- [Evaluasi](#evaluasi)
- [Cara Menjalankan Proyek](#cara-menjalankan-proyek)

## Pendahuluan
Di era digital, anime telah menjadi salah satu bentuk hiburan paling populer secara global. Dengan ribuan judul yang tersedia, pengguna sering menghadapi *overload informasi*, kesulitan menemukan anime yang sesuai dengan selera mereka. Proyek ini bertujuan membangun sistem rekomendasi anime yang personal dan efektif untuk membantu pengguna menemukan anime relevan berdasarkan preferensi mereka, baik dari karakteristik anime (genre, tipe) maupun perilaku tontonan pengguna lain. Sistem ini meningkatkan pengalaman pengguna di platform streaming atau basis data anime, meningkatkan *engagement* dan kepuasan pengguna.

**Referensi**:  
[1] J. L. Herlocker, J. A. Konstan, A. Borchers, and J. Riedl, "An algorithmic framework for performing collaborative filtering," in *Proceedings of the 22nd annual international ACM SIGIR conference on Research and development in information retrieval*, 1999, pp. 230-237.

## Pemahaman Bisnis

### Pernyataan Masalah
- **Overload Informasi**: Pengguna kesulitan menemukan anime baru di tengah banyaknya pilihan, menyebabkan kelelahan keputusan dan mengurangi kepuasan.
- **Kurangnya Personalisasi**: Banyak platform gagal memberikan rekomendasi yang sangat personal, sehingga pengguna mungkin beralih ke platform lain.

### Tujuan Proyek
- **Mempermudah Penemuan Konten**: Merekomendasikan anime relevan secara efisien, mengurangi waktu pencarian.
- **Meningkatkan Kepuasan Pengguna**: Menyediakan rekomendasi personal dan akurat untuk pengalaman pengguna yang lebih baik.

### Pendekatan Solusi
Proyek ini mengimplementasikan dua pendekatan:
1. **Content-based Filtering**: Merekomendasikan anime berdasarkan kemiripan atribut (genre, tipe) menggunakan *Cosine Similarity* pada fitur genre yang diubah ke representasi numerik dengan *TF-IDF*.
2. **Collaborative Filtering**: Merekomendasikan anime berdasarkan pola rating pengguna serupa menggunakan *Singular Value Decomposition (SVD)*.

## Pemahaman Data
Dataset yang digunakan adalah **Anime Recommendations Database** dari Kaggle: [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Terdiri dari dua file CSV:
- **anime.csv**:
  - Jumlah data: 12,294 entri.
  - Kolom: `anime_id` (int64), `name` (object), `genre` (object, 62 null), `type` (object, 25 null), `episodes` (object), `rating` (float64, 230 null), `members` (int64).
- **rating.csv**:
  - Jumlah data: 7,813,737 entri.
  - Kolom: `user_id` (int64), `anime_id` (int64), `rating` (int64, -1 hingga 10, -1 menunjukkan "ditonton tapi tidak dinilai").

**Insight dari Eksplorasi Data**:
- **Jumlah Genre Unik**: 43 genre.
- **Tipe Anime**:
  - TV: 3,787
  - OVA: 3,311
  - Movie: 2,348
  - Special: 1,676
  - ONA: 659
  - Music: 488
- **Distribusi Rating Pengguna** (tanpa -1): Puncak di rating 7, 8, 9, dengan rating 8 paling sering. Rating rendah (1-5) jarang, menunjukkan kecenderungan pengguna memberikan rating tinggi.
- **Penggabungan Awal**: Gabungan `rating_df` dan `anime_df` menghasilkan 7,813,727 baris dan 9 kolom.

## Persiapan Data
1. **Penanganan Missing Values**:
   - Kolom `genre` (62 null) diisi dengan 'Unknown'.
   - Kolom `rating` (230 null) diisi dengan rata-rata rating.
   - **Alasan**: Mencegah error dan bias pada model.
2. **Penanganan Rating -1**:
   - Baris dengan rating -1 dihapus (sisa 6,337,239 baris).
   - **Alasan**: Rating -1 tidak memberikan informasi preferensi untuk *collaborative filtering*.
3. **Penanganan Duplikat**:
   - Duplikat berdasarkan `anime_id` dihapus (tidak ada duplikat ditemukan).
   - **Alasan**: Menghindari redundansi.
4. **Penggabungan Dataset**:
   - `rating_df` dan `anime_df` digabung berdasarkan `anime_id`, menghasilkan `merged_df` (6,337,239 baris, 9 kolom).
5. **Pembersihan Kolom Genre**:
   - Ganti koma dengan spasi pada kolom `genre` dan simpan di `genre_cleaned`.
   - **Alasan**: Memastikan *TfidfVectorizer* memproses genre sebagai token terpisah.

## Pemodelan
### Content-based Filtering
- **Algoritma**: *Cosine Similarity*.
- **Cara Kerja**:
  - *TF-IDF Vectorization*: Kolom `genre_cleaned` diubah menjadi matriks *TF-IDF* (12,294 anime, 43 fitur genre).
  - *Cosine Similarity*: Matriks kemiripan kosinus dihitung untuk setiap pasangan anime (12,294 x 12,294).
  - **Rekomendasi**: Anime dengan skor kemiripan tertinggi direkomendasikan.
- **Contoh Rekomendasi (Death Note)**:
  ```plaintext
  name                                              genre     type  rating  similarity_score
  Death Note Rewrite  Mystery, Police, Psychological, Supernatural, ...  Special 7.84 1.000000
  Mousou Dairinin     Drama, Mystery, Police, Psychological, Superna...  TV      7.74 0.967703
  Higurashi no Naku Koro ni Kai Mystery, Psychological, Supernatural, Thriller  TV      8.41 0.879514
  ...
  ```
- **Kelebihan**:
  - Tidak memerlukan data rating pengguna lain (*cold-start* friendly).
  - Rekomendasi transparan dan mudah dijelaskan.
  - Merekomendasikan item yang belum di-rating.
- **Kekurangan**:
  - Over-specialization, kurang beragam.
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

## Evaluasi
### Metrik Evaluasi
1. **RMSE (Root Mean Squared Error)**:
   - Mengukur rata-rata magnitudo error prediksi rating.
   - Formula: `RMSE = sqrt(1/N * Σ(yi - ŷi)²)`
2. **MAE (Mean Absolute Error)**:
   - Mengukur rata-rata nilai absolut error prediksi.
   - Formula: `MAE = 1/N * Σ|yi - ŷi|`

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

**Kesimpulan Umum**: SVD memberikan prediksi rating akurat, sementara *Content-based Filtering* menghasilkan rekomendasi tematik yang relevan. Kedua pendekatan saling melengkapi.

## Cara Menjalankan Proyek
1. **Unduh Dataset**:
   - Unduh `anime.csv` dan `rating.csv` dari [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).
2. **Siapkan Lingkungan Python**:
   - Gunakan Python 3.7+.
   - Instal library:
     ```bash
     pip install pandas numpy scikit-learn scikit-surprise matplotlib seaborn
     ```
3. **Tempatkan File Dataset**:
   - **Google Colab**: Unggah file ke Google Drive (misalnya, folder `dataset_anime`), sesuaikan `data_path` di kode.
   - **Lokal**: Simpan file CSV di direktori yang sama dengan file `.ipynb`, atau sesuaikan `data_path`.
4. **Jalankan Jupyter Notebook**:
   - Buka Jupyter Notebook/JupyterLab.
   - Navigasi ke direktori proyek.
   - Buka file `.ipynb` dan jalankan setiap cell secara berurutan.
```

**Catatan**:
- File `README.md` ini mengikuti struktur yang diminta, dengan penyesuaian untuk kejelasan dan format Markdown yang rapi.
- Pastikan semua link (misalnya, Kaggle) aktif dan sesuai saat digunakan.
- Jika ada visualisasi seperti "Distribution of Anime Ratings", Anda dapat menambahkan gambar ke folder proyek dan menyisipkannya di README dengan sintaks `![Distribution](path/to/image.png)`. Namun, karena tidak ada file gambar yang disediakan, bagian ini tidak dimasukkan.