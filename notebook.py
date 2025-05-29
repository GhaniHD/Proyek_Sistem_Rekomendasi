#!/usr/bin/env python
# coding: utf-8

# ## Import library

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Loading

# In[4]:


anime_df = pd.read_csv('dataset/anime.csv')
rating_df = pd.read_csv('dataset/rating.csv')
print("Datasets loaded successfully!")
print(f"Shape of anime_df: {anime_df.shape}")
print(f"Shape of rating_df: {rating_df.shape}")


# ## Data Understanding

# In[5]:


print("\n--- Anime DataFrame Info ---")
anime_df.info()
print("\nFirst 5 rows of anime_df:")
print(anime_df.head())
print("\nMissing values in anime_df:")
print(anime_df.isnull().sum())
print("\nNumber of unique genres:")


# Memisahkan genre dan menghitung genre yang unik

# In[6]:


all_genres = anime_df['genre'].dropna().str.split(', ').explode().unique()
print(len(all_genres))
print("\nTop 10 most frequent anime types:")
print(anime_df['type'].value_counts().head(10))


# Rating DataFrame Info

# In[7]:


rating_df.info()
print("\nFirst 5 rows of rating_df:")
print(rating_df.head())
print("\nMissing values in rating_df:")
print(rating_df.isnull().sum())
print("\nDistribution of ratings (excluding -1):")
print(rating_df[rating_df['rating'] != -1]['rating'].value_counts().sort_index())


# Visualisasi distribusi rating

# In[8]:


plt.figure(figsize=(10, 6))
sns.histplot(rating_df[rating_df['rating'] != -1]['rating'], bins=10, kde=True)
plt.title('Distribution of Anime Ratings (excluding -1)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# Visualisasi top N genres

# In[9]:


genre_counts = anime_df['genre'].dropna().str.split(', ').explode().value_counts()
plt.figure(figsize=(12, 7))
sns.barplot(x=genre_counts.head(15).index, y=genre_counts.head(15).values)
plt.title('Top 15 Most Frequent Anime Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Tinjauan Dataframe Gabungan (setelah inspeksi awal)

# In[10]:


# Merge datasets early to check for potential issues with merging
initial_merged_df = pd.merge(rating_df, anime_df, on='anime_id', suffixes=('_user', '_anime'))
print(f"Shape of initial_merged_df: {initial_merged_df.shape}")
print("Initial Merged DataFrame Head:")
print(initial_merged_df.head())


# ## Data Preparation

# Pra-pemrosesan DataFrame Anime

# In[11]:


print("Menangani nilai yang hilang pada anime_df...")


# Mengisi nilai 'genre' yang kosong dengan 'Tidak Diketahui'

# In[12]:


anime_df['genre'] = anime_df['genre'].fillna('Unknown')


# Mengisi nilai 'rating' (penilaian keseluruhan anime) yang kosong dengan rata-rata

# In[13]:


anime_df['rating'] = anime_df['rating'].fillna(anime_df['rating'].mean())
print("Jumlah nilai yang hilang dalam anime_df setelah pengisian:")
print(anime_df.isnull().sum())


# Menghapus data duplikat berdasarkan 'anime_id' untuk memastikan setiap anime unik

# In[14]:


initial_anime_rows = anime_df.shape[0]
anime_df.drop_duplicates(subset='anime_id', inplace=True)
print(f"Dropped {initial_anime_rows - anime_df.shape[0]} duplicate anime entries.")
print(f"Shape of anime_df after dropping duplicates: {anime_df.shape}")


# Pra-pemrosesan DataFrame Rating
# 
# Menghapus rating -1: ini menunjukkan anime sudah ditonton tapi belum diberi nilai,
# 
# yang tidak berguna untuk prediksi rating eksplisit dalam collaborative filtering.

# In[16]:


initial_rating_rows = rating_df.shape[0]
rating_df = rating_df[rating_df['rating'] != -1]
print(f"Removed {initial_rating_rows - rating_df.shape[0]} ratings of -1.")
print(f"Shape of rating_df after removing -1 ratings: {rating_df.shape}")


# Menggabungkan dataset

# In[17]:


print("\nMerging anime and rating dataframes...")
merged_df = pd.merge(rating_df, anime_df, on='anime_id', suffixes=('_user', '_anime'))
print(f"Shape of merged_df after merging: {merged_df.shape}")
print("\nMerged DataFrame Head (after preprocessing):")
print(merged_df.head())


# Membersihkan kolom genre untuk TF-IDF (menghapus koma dan spasi berlebih)

# In[18]:


anime_df['genre_cleaned'] = anime_df['genre'].apply(lambda x: ' '.join(x.replace(',', ' ').split()))
print("\n'genre_cleaned' column created for Content-based Filtering.")


# ## Modeling

# Inisialisasi TF-IDF Vectorizer
# 
# stop_words='english' menghapus kata-kata umum bahasa Inggris yang kurang bermakna

# In[19]:


tfidf = TfidfVectorizer(stop_words='english')


# Melatih dan mentransformasikan kolom 'genre_cleaned' untuk membuat matriks TF-IDF

# In[20]:


tfidf_matrix = tfidf.fit_transform(anime_df['genre_cleaned'])


# In[21]:


print(f"Ukuran matriks TF-IDF: {tfidf_matrix.shape}")


# Menghitung matriks kemiripan kosinus
# 
# Matriks ini menunjukkan kemiripan antar anime berdasarkan vektor genre mereka

# In[22]:


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Ukuran matriks Kemiripan Kosinus: {cosine_sim.shape}")


# Membuat Series yang memetakan nama anime ke indeksnya di dataframe
# 
# Ini memudahkan pencarian indeks anime berdasarkan namanya

# In[23]:


indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()


# Fungsi untuk mendapatkan rekomendasi berbasis konten 
# 
# Mendapatkan indeks anime yang sesuai dengan judul
# 
# Mendapatkan skor kemiripan pasangan semua anime dengan anime tersebut
# 
# Mengurutkan anime berdasarkan skor kemiripan
# 
# Mendapatkan skor untuk top N anime yang paling mirip (tidak termasuk dirinya sendiri)
# 

# In[27]:


def get_content_recommendations(title, cosine_sim_matrix=cosine_sim, df=anime_df, indices_series=indices, top_n=10):
    """
    Generates content-based recommendations for a given anime title.

    Args:
        title (str): The name of the anime to get recommendations for.
        cosine_sim_matrix (np.array): The cosine similarity matrix of anime.
        df (pd.DataFrame): The anime dataframe.
        indices_series (pd.Series): A Series mapping anime names to their indices.
        top_n (int): The number of top recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended anime details and similarity scores.
    """
    if title not in indices_series:
        print(f"Anime '{title}' not found in the database. Please check the spelling.")
        return pd.DataFrame() 


    idx = indices_series[title]

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1] 

    anime_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[anime_indices][['name', 'genre', 'type', 'rating']].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return recommendations


# Contoh Rekomendasi Berbasis Konten

# In[28]:


print("\nExample Content-based Recommendations for 'Death Note':")
content_recs = get_content_recommendations('Death Note')
if not content_recs.empty:
    print(content_recs)


# In[29]:


print("\nExample Content-based Recommendations for 'Naruto':")
content_recs_naruto = get_content_recommendations('Naruto')
if not content_recs_naruto.empty:
    print(content_recs_naruto)


# ## Collaborative Filtering

# Pendekatan ini merekomendasikan anime berdasarkan perilaku pengguna lain yang memiliki preferensi serupa. Kita akan menggunakan library Surprise dan algoritma Singular Value Decomposition (SVD) untuk memprediksi rating yang belum diberikan.

#  Pemodelan (lanjutan) - Collaborative Filtering 

# Persiapkan data untuk library Surprise
# 
# Objek Reader mendefinisikan skala rating (1 sampai 10 di dataset ini)
# 

# In[33]:


reader = Reader(rating_scale=(1, 10))


# Muat data ke format Dataset Surprise dari merged_df
# 
# Kita hanya butuh user_id, anime_id, dan rating untuk collaborative filtering

# In[37]:


data = Dataset.load_from_df(merged_df[['user_id', 'anime_id', 'rating_user']], reader)


# Bangun trainset dari seluruh data untuk melatih model akhir Untuk evaluasi biasanya kita pisah train/test, tapi untuk prediksi ke semua userkita bisa latih dengan seluruh data.
# 
# Atau untuk evaluasi cross-validation, kita bisa gunakan `cross_validate` nanti.

# In[38]:


trainset = data.build_full_trainset()


# Gunakan algoritma SVD (Singular Value Decomposition)
# 
# SVD adalah teknik faktorisasi matriks yang sering dipakai di collaborative filtering

# In[39]:


algo = SVD(random_state=42)


# Latih model menggunakan seluruh trainset

# In[40]:


print("Melatih model SVD untuk Collaborative Filtering...")
algo.fit(trainset)
print("Model SVD berhasil dilatih!")


# Fungsi get_collaborative_recommendations ini digunakan untuk menghasilkan rekomendasi anime bagi seorang pengguna tertentu dengan menggunakan metode collaborative filtering berbasis model SVD yang sudah dilatih; fungsi ini pertama-tama mengambil semua anime yang tersedia dan menghapus anime yang sudah pernah dirating oleh pengguna tersebut agar hanya merekomendasikan anime baru, kemudian memprediksi rating estimasi untuk setiap anime yang belum dirating oleh pengguna tersebut menggunakan model SVD, mengurutkan anime berdasarkan prediksi rating tertinggi, dan mengembalikan DataFrame yang berisi detail anime teratas beserta prediksi rating khusus untuk pengguna tersebut.

# In[41]:


def get_collaborative_recommendations(user_id, num_recommendations=10, df_anime=anime_df, df_merged=merged_df, svd_model=algo):
    """
    Menghasilkan rekomendasi collaborative filtering untuk user tertentu.

    Argumen:
        user_id (int): ID user yang ingin direkomendasikan.
        num_recommendations (int): Jumlah rekomendasi teratas yang dikembalikan.
        df_anime (pd.DataFrame): Dataframe anime.
        df_merged (pd.DataFrame): Dataframe gabungan rating dan info anime.
        svd_model: Model SVD yang sudah dilatih.

    Mengembalikan:
        pd.DataFrame: DataFrame berisi detail anime rekomendasi dan prediksi rating.
    """

    all_anime_ids = df_anime['anime_id'].unique()

    rated_anime_ids = df_merged[df_merged['user_id'] == user_id]['anime_id'].unique()

    anime_to_predict = [anime_id for anime_id in all_anime_ids if anime_id not in rated_anime_ids]

    predictions = []
    for anime_id in anime_to_predict:
        pred = svd_model.predict(uid=user_id, iid=anime_id)
        predictions.append((anime_id, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_anime_ids = [x[0] for x in predictions[:num_recommendations]]
    top_estimated_ratings = [x[1] for x in predictions[:num_recommendations]]

    recommended_anime_details = df_anime[df_anime['anime_id'].isin(top_anime_ids)][['anime_id', 'name', 'genre', 'type', 'rating']].copy()

    final_recommendations = []
    for i, anime_id in enumerate(top_anime_ids):
        anime_info = recommended_anime_details[recommended_anime_details['anime_id'] == anime_id].iloc[0]
        final_recommendations.append({
            'anime_id': anime_info['anime_id'],
            'name': anime_info['name'],
            'genre': anime_info['genre'],
            'type': anime_info['type'],
            'average_anime_rating': anime_info['rating'],
            'predicted_rating_for_user': top_estimated_ratings[i]
        })
    return pd.DataFrame(final_recommendations)


# Contoh rekomendasi Collaborative Filtering untuk seorang user
# 
# Cari user_id yang ada di dataset (misal user yang sudah memberi rating anime)
# 
# Supaya pasti dapat user yang sudah punya rating, kita filter merged_df

# In[42]:


users_with_ratings = merged_df['user_id'].unique()
if len(users_with_ratings) > 0:
    example_user_id = np.random.choice(users_with_ratings)
else:
    example_user_id = 1

print(f"\nContoh Rekomendasi Collaborative Filtering untuk user_id: {example_user_id}:")
cf_recs = get_collaborative_recommendations(example_user_id)
print(cf_recs)


# Untuk melihat anime apa saja yang sudah dirating user tersebut:

# In[46]:


print(f"\nAnime rated by user_id {example_user_id}:")
user_rated_anime = merged_df[merged_df['user_id'] == example_user_id][['name', 'rating_user', 'genre']].sort_values(by='rating_user', ascending=False)
print(user_rated_anime.head(10))


# ##  Evaluation

# Lakukan cross-validation 5-fold untuk mengukur performa model SVD

# In[47]:


print("Melakukan cross-validation 5-fold untuk model SVD...")
cross_validation_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# RMSE (Root Mean Squared Error): Mengukur besarnya rata-rata kesalahan.
# 
# RMSE yang lebih rendah menunjukkan akurasi yang lebih baik dalam memprediksi peringkat.

# In[48]:


print(f"Mean RMSE: {cross_validation_results['test_rmse'].mean():.4f}")
print(f"Standard Deviation of RMSE: {cross_validation_results['test_rmse'].std():.4f}")


# MAE (Mean Absolute Error): Mengukur rata-rata perbedaan absolut antara peringkat yang diprediksi dan aktual.
# 
# Mirip dengan RMSE, MAE yang lebih rendah menunjukkan akurasi yang lebih tinggi.

# In[49]:


print(f"Mean MAE: {cross_validation_results['test_mae'].mean():.4f}")
print(f"Standard Deviation of MAE: {cross_validation_results['test_mae'].std():.4f}")


# Tampilkan kembali contoh untuk penilaian kualitatif

# In[50]:


content_recs_eval = get_content_recommendations('Death Note')
if not content_recs_eval.empty:
    print(content_recs_eval)
    print("\nObservation: The recommendations for 'Death Note' (a psychological thriller) tend to be other anime with 'Thriller' or 'Mystery' genres, which indicates good content similarity.")

print("\n--- Conclusion of Evaluation ---")
print("The SVD model for Collaborative Filtering shows a reasonable RMSE, indicating good predictive accuracy for user ratings.")
print("Content-based filtering provides recommendations based on clear genre similarity, which is intuitively logical.")

