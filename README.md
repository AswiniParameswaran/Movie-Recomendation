# 🎬  Movie Recommendation System

This project is a Python-based movie recommender that suggests similar movies based on the content of a given movie. It uses natural language processing and machine learning techniques like **TF-IDF vectorization** and **cosine similarity** to find relationships between movies based on their **genres, keywords, cast, tagline, and director**.

---

## 🔍 Features

- Content-based filtering using textual data
- Handles typos or close matches in movie names
- Recommends top 29 movies similar to a selected one
- Uses cosine similarity to measure movie similarity
- Supports visual inspection using heatmaps (optional)

---

## 📁 Dataset

The dataset contains movie metadata and is typically sourced from:
- The Movie Database (TMDb)
- [Kaggle Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

It includes fields like:
- `title`
- `genres`
- `keywords`
- `cast`
- `director`
- `tagline`
- `overview` *(optional)*

---

## 📦 Dependencies

Make sure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
