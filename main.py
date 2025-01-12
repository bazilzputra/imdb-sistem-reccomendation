import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Fungsi untuk membersihkan teks
def clean_text(text):
    clean_spcl = re.compile('[/(){}\[\]\|@,;]')
    clean_symbol = re.compile('[^0-9a-z #+_]')

    text = text.lower()  # Lowercase
    text = clean_spcl.sub(' ', text)  # Hapus karakter spesial
    text = clean_symbol.sub(' ', text)  # Hapus simbol

    stop_factory = StopWordRemoverFactory()
    stopword_remover = stop_factory.create_stop_word_remover()
    text = stopword_remover.remove(text)  # Hapus stopword

    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    text = stemmer.stem(text)  # Stemming

    return text

# Load dataset
df = pd.read_csv("IMDb_Top_250_Cleaned.csv")

# Bersihkan kolom Title
df['Title_Clean'] = df['Title'].apply(clean_text)

# TF-IDF dan Cosine Similarity
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.0)
tfidf_matrix = tf.fit_transform(df['Title_Clean'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Setup Streamlit
st.title("Sistem Rekomendasi Film IMDb")
st.write("Cari rekomendasi film berdasarkan judul!")

# Input dan rekomendasi
film_input = st.text_input("Masukkan judul film")
tombol_rekomendasi = st.button("Tampilkan Rekomendasi")

if tombol_rekomendasi:
    if film_input.strip():
        indices = pd.Series(df.index, index=df['Title'])
        matching_indices = indices[indices.index.str.contains(film_input, case=False, na=False)]

        if not matching_indices.empty:
            idx = matching_indices.iloc[0]
            score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)

            # Rekomendasi 10 film teratas
            top_recommendations = score_series.iloc[1:11].index  # Skip indeks 0 (film input)
            recommended_movies = df.iloc[top_recommendations][['Title', 'Year', 'IMDb Rating']]

            st.write("Rekomendasi film berdasarkan input Anda:")
            st.dataframe(recommended_movies)
        else:
            st.write("Maaf, film tidak ditemukan!")
    else:
        st.write("Harap masukkan judul film.")

