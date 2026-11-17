import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies():
    """Load the movie dataset and prepare text for comparison."""
    movies = pd.read_csv("movies.csv")
    movies["combined_features"] = (
        movies["genres"].fillna("") + " " +
        movies["description"].fillna("")
    )
    return movies


def build_similarity_matrix(movies):
    """Convert movie text into TF-IDF vectors and compare them."""
    vectorizer = TfidfVectorizer(stop_words="english")
    feature_matrix = vectorizer.fit_transform(movies["combined_features"])
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix


def recommend_movies(movie_name, movies, similarity_matrix, number=5):
    """Return the most similar movies for the selected movie."""
    matching_movies = movies[
        movies["title"].str.casefold() == movie_name.strip().casefold()
    ]

    if matching_movies.empty:
        return None

    movie_index = matching_movies.index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similarity_scores = sorted(
        similarity_scores, key=lambda item: item[1], reverse=True
    )

    # Skip the selected movie itself.
    recommendations = similarity_scores[1:number + 1]
    recommended_indices = [index for index, score in recommendations]

    return movies.iloc[recommended_indices][["title", "genres"]]


def main():
    print("=" * 45)
    print("       MOVIE RECOMMENDATION SYSTEM")
    print("=" * 45)

    movies = load_movies()
    similarity_matrix = build_similarity_matrix(movies)

    print("\nAvailable movies:")
    for title in movies["title"]:
        print("-", title)

    movie_name = input("\nEnter a movie you like: ")
    recommendations = recommend_movies(movie_name, movies, similarity_matrix)

    if recommendations is None:
        print("\nMovie not found. Please enter a title from the list.")
        return

    print("\nMovies recommended for you:")
    for number, (_, movie) in enumerate(recommendations.iterrows(), start=1):
        print(f"{number}. {movie['title']} ({movie['genres']})")


if __name__ == "__main__":
    main()
