import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# Global variables
sub_replace = re.compile("[^0-9a-z #+_]")
stopwords_set = set(stopwords.words("english"))


def get_top_n_words(corpus, n=None):
    """Analyze word frequencies in the corpus"""
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def clean_txt(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = sub_replace.sub("", text)
    text = " ".join(word for word in text.split() if word not in stopwords_set)
    return text


def calculate_similarity_matrix(df):
    """Calculate TF-IDF and cosine similarity matrix"""
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), stop_words="english")
    tfidf_matrix = tf.fit_transform(df["desc_clean"])
    return linear_kernel(tfidf_matrix, tfidf_matrix)


def recommendations(name, cosine_similarity, df, indices):
    """Generate recommendations for a given hotel"""
    try:
        idx = indices[indices == name].index[0]
        score_series = pd.Series(cosine_similarity[idx]).sort_values(ascending=False)
        top_10_indexes = list(score_series[1:11].index)
        recommended_hotels = [list(df.index)[i] for i in top_10_indexes]
        return recommended_hotels
    except Exception as e:
        print(f"Error generating recommendations for {name}: {str(e)}")
        return []


def evaluate_recommendations(model_recommendations_df, actual_recommendations_df, df):
    """Evaluate recommendation model against actual recommendations"""
    recommended_hotels = set()
    for cols in model_recommendations_df.iloc[:, :].values:
        recommended_hotels.update(cols)

    coverage = len(recommended_hotels) / len(df.index)

    total_precision = 0
    total_recall = 0
    count = 0

    for hotel in df.index:
        if (
            hotel in model_recommendations_df.index
            and hotel in actual_recommendations_df.index
        ):
            model_recs = set(model_recommendations_df.loc[hotel].tolist())
            actual_recs = set(actual_recommendations_df.loc[hotel].tolist())

            correct_predictions = len(model_recs.intersection(actual_recs))

            precision = correct_predictions / len(model_recs)
            recall = correct_predictions / len(actual_recs)

            total_precision += precision
            total_recall += recall
            count += 1

    avg_precision = total_precision / count if count > 0 else 0
    avg_recall = total_recall / count if count > 0 else 0

    f1_score = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    # Print results
    print("\n=== Recommendation System Evaluation ===")
    print(f"\nCoverage Analysis:")
    print(f"Total number of hotels: {len(df.index)}")
    print(f"Number of recommended hotels: {len(recommended_hotels)}")
    print(f"Coverage: {coverage:.3f}")

    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")

    return {
        "coverage": coverage,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": f1_score,
    }


def print_hotel_metrics(
    hotel_name, model_recommendations_df, actual_recommendations_df
):
    """Compare recommendations for a specific hotel"""
    if (
        hotel_name in model_recommendations_df.index
        and hotel_name in actual_recommendations_df.index
    ):
        model_recs = set(model_recommendations_df.loc[hotel_name].tolist())
        actual_recs = set(actual_recommendations_df.loc[hotel_name].tolist())

        common_recs = model_recs.intersection(actual_recs)

        precision = len(common_recs) / len(model_recs)
        recall = len(common_recs) / len(actual_recs)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\nRecommendation Analysis for Hotel '{hotel_name}':")
        print(f"Model Recommendations: {list(model_recs)[:3]}...")
        print(f"Actual Recommendations: {list(actual_recs)[:3]}...")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Number of common recommendations: {len(common_recs)}")


def main():
    """Main function"""
    print("Loading and processing data...")
    df = pd.read_csv("Seattle_Hotels.csv", encoding="latin-1")
    df.set_index("name", inplace=True)

    print("Cleaning text descriptions...")
    df["desc_clean"] = df["desc"].apply(clean_txt)

    print("Calculating similarity matrix...")
    tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), stop_words="english")
    tfidf_matrix = tf.fit_transform(df["desc_clean"])
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index)

    print("Generating model recommendations...")
    model_recommendations = {}
    for hotel in df.index:
        model_recommendations[hotel] = recommendations(
            hotel, cosine_similarity, df, indices
        )

    model_rec_df = pd.DataFrame.from_dict(model_recommendations, orient="index")
    model_rec_df.columns = [f"Recommendation {i+1}" for i in range(10)]

    print("Loading actual recommendations...")
    actual_recommendations_df = pd.read_csv(
        "Seattle_Hotels_Recommendations.csv", index_col=0
    )

    print("\nEvaluating recommendations...")
    metrics = evaluate_recommendations(model_rec_df, actual_recommendations_df, df)

    sample_hotels = ["The Loyal Inn", "Hilton Seattle", "Kimpton Hotel Vintage Seattle"]
    print("\n=== Sample Recommendations Comparison ===")
    for hotel in sample_hotels:
        print_hotel_metrics(hotel, model_rec_df, actual_recommendations_df)

    print("\n=== Summary ===")
    print(f"Coverage: {metrics['coverage']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")


if __name__ == "__main__":
    main()
