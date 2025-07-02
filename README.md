
# Movie Recommendation System

## Project Report

### Introduction
In today’s digital age, users face an overwhelming amount of movie content, making it difficult to find films that match their preferences. This project develops a **Hybrid Movie Recommendation System** using Python and the MovieLens dataset. By integrating popularity-based, content-based, collaborative filtering, and matrix factorization techniques, the system delivers personalized movie suggestions to enhance user engagement and satisfaction, suitable for streaming platforms.

---

## Table of Contents
1. [Datasets Overview](#datasets-overview)
2. [Problem Statement](#problem-statement)
3. [Goal](#goal)
4. [Methodology](#methodology)
5. [Implementation](#implementation)
6. [Key Insights](#key-insights)
7. [Conclusion](#conclusion)
8. [Installation and Usage](#installation-and-usage)
9. [Future Improvements](#future-improvements)

---

## Datasets Overview
This project utilizes the **MovieLens dataset**, a widely recognized benchmark for recommendation systems, provided by GroupLens. The dataset includes:

1. **`movies.csv`**: Movie metadata with columns `movieId`, `title`, and `genres`.
2. **`ratings.csv`**: User ratings with `userId`, `movieId`, `rating` (0.5–5.0 scale), and `timestamp`.
3. **`tags.csv`**: User-generated tags with `userId`, `movieId`, `tag`, and `timestamp`.
4. **`links.csv`**: Links to external databases (IMDb, TMDb) with `movieId`, `imdbId`, and `tmdbId`.

**Download Reference**: The small MovieLens dataset can be obtained from [GroupLens](https://grouplens.org/datasets/movielens/). This project uses the small version for efficiency, but the approach is scalable to larger datasets.

---

## Problem Statement
Users struggle to find movies tailored to their tastes amidst digital content overload. The objective is to build a recommendation system that understands user preferences and provides personalized suggestions to improve engagement and satisfaction.

---

## Goal
To create a hybrid movie recommendation engine that:
- Suggests movies based on user ratings.
- Incorporates genre similarities and user behavior.
- Is scalable and interpretable.
- Can be integrated into real-world applications like streaming platforms.

---

## Methodology
The project follows a structured approach, starting with data loading and preprocessing, followed by implementing multiple recommendation techniques—popularity-based, content-based, collaborative filtering (KNN), and matrix factorization (SVD)—and culminating in a hybrid model combining content-based and collaborative methods.

---

## Implementation

### Step 1: Import Libraries
Imported essential libraries (`pandas`, `numpy`, `scikit-learn`, `surprise`) for data processing, similarity computation, and recommendation modeling.

### Step 2: Load Datasets
Loaded the MovieLens dataset files (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) into Pandas DataFrames.

### Step 3: Merge Ratings and Movies
Merged the `ratings` and `movies` DataFrames to create a unified dataset combining ratings and movie metadata.

### Step 4: Calculate Popularity Metrics
Calculated the average rating and number of ratings per movie to establish popularity metrics.

### Step 5: Filter Popular Movies
Filtered movies with at least 50 ratings to exclude obscure or biased entries.

### Step 6: Sort by Rating
Sorted the filtered movies by average rating to identify the top 10 most popular movies.

### Step 7: Content-Based Filtering Setup
Initiated content-based filtering by preparing to use movie genres for similarity analysis.

### Step 8: Handle Missing Genres
Filled missing genre values with empty strings to ensure compatibility with further processing.

### Step 9: TF-IDF Vectorization
Applied TF-IDF vectorization to encode movie genres into a numerical matrix for similarity computation.

### Step 10: Cosine Similarity
Computed cosine similarity between all movies based on their genre TF-IDF vectors.

### Step 11: Reverse Index
Created a reverse index mapping movie titles to DataFrame indices for efficient lookup.

### Step 12: Recommendation Function
Developed a function to recommend the top 10 movies similar to a given title based on genre similarity.

### Step 13: Output Results
Displayed the top 10 popular movies and sample content-based recommendations for "Fight Club (1999)".

### Step 14: Example Usage
Showed content-based recommendations for "Nixon (1995)" as an example.

### Step 15: Import Additional Libraries
Added libraries for collaborative filtering (KNN) and matrix factorization (SVD) to support the hybrid model.

### Step 16: Content-Based Recommender
Reimplemented the content-based recommender with a refined function name for clarity in the hybrid context.

### Step 17: Collaborative Filtering (KNN)
Implemented user-based collaborative filtering using KNN to recommend movies based on similar users’ ratings.

### Step 18: Matrix Factorization (SVD)
Applied SVD for matrix factorization, training the model on ratings and generating recommendations based on predicted scores.

### Step 19: Hybrid Recommender
Built a hybrid recommender combining genre similarity (content-based) with SVD-predicted ratings (collaborative).

### Step 20: Example Outputs
Demonstrated outputs for all recommenders: content-based for "Toy Story (1995)", collaborative (KNN) and SVD for user 5, and hybrid for user 5 based on "Inception (2010)".

---

## Key Insights
- **Popularity-Based**: Identified top movies like "Shawshank Redemption" based on ratings, effective for new users but not personalized.
- **Content-Based**: Successfully recommended genre-similar movies (e.g., "Toy Story 2" for "Toy Story"), limited to genre features.
- **Collaborative (KNN)**: Suggested diverse movies (e.g., "Jurassic Park") based on similar users, effective but sensitive to sparse data.
- **SVD**: Recommended high-quality films (e.g., "Princess Bride") with an RMSE of ~0.88, balancing accuracy and scalability.
- **Hybrid**: Combined genre similarity with user preferences (e.g., "Dark Knight" for "Inception" fans), offering the most robust and personalized results.

---

## Conclusion
This project successfully developed a scalable and interpretable movie recommendation system using a hybrid approach. By integrating popularity-based, content-based, collaborative filtering, and matrix factorization techniques, it delivers personalized suggestions ready for integration into streaming platforms. The system is extensible, with potential for enhancements like tag-based filtering or real-time updates.

---

## Installation and Usage
### Prerequisites
- Python 3.11+
- Required libraries:
  ```bash
  pip install pandas numpy scikit-learn scikit-surprise
  ```

### Running the Project
1. **Download MovieLens Dataset**: Obtain the small dataset from [GroupLens](https://grouplens.org/datasets/movielens/).
2. **Place Files**: Ensure `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv` are in the working directory.
3. **Run the Notebook**:
   ```bash
   jupyter notebook movie_recommendation.ipynb
   ```
   Or convert to a script:
   ```bash
   jupyter nbconvert --to script movie_recommendation.ipynb
   python movie_recommendation.py
   ```

---

## Future Improvements
- **Tag Integration**: Incorporate `tags.csv` for enhanced content-based filtering.
- **Hyperparameter Tuning**: Optimize KNN neighbors or SVD factors for better performance.
- **Dynamic Thresholds**: Adjust the 50-rating filter dynamically based on dataset size.
- **API Deployment**: Develop a Flask/Django API for real-time recommendations.
- **Enhanced Evaluation**: Include precision/recall metrics alongside RMSE for a fuller assessment.

