import re
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud
from pathlib import Path

st.set_page_config(page_title="Sentiment Analysis NLP App", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_lr.joblib"
DATA_PATH = BASE_DIR / "data" / "rockygerung_predicted.csv"  # kolom: full_text, sentiment
EXTRA_STOPWORDS = ["ya", "yg", "ga", "yuk", "dah", "ngga", "engga", "ygy"]
SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_COLORS = {"positive": "#2e9e4f", "neutral": "#4c78a8", "negative": "#d64545"}


# ---------- Loader (di-cache agar tidak dimuat ulang tiap interaksi) ----------
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["full_text"]).copy()
    df["full_text"] = df["full_text"].astype(str)
    return df


@st.cache_data
def get_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
    except Exception:
        words = set()
    words.update(EXTRA_STOPWORDS)
    return sorted(words)


# ---------- Preprocessing (sama dengan cleansing di notebook) ----------
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r"http\S+", "", string)
    string = re.sub(r"@\w+", "", string)
    string = re.sub(r"#\w+", "", string)
    string = re.sub(r"\d+", "", string)
    string = re.sub(r"[^\w\s]", "", string)
    string = re.sub(r"\s+", " ", string).strip()
    return string


# ---------- Analisis sentimen ----------
def analyze_token_sentiment(text, pipeline):
    result = {"positives": [], "negatives": [], "neutral": []}
    has_proba = hasattr(pipeline, "predict_proba")

    for word in text.split():
        pred = pipeline.predict([word])[0]
        proba = round(float(pipeline.predict_proba([word]).max()), 4) if has_proba else None
        item = [word, proba]

        if pred == "positive":
            result["positives"].append(item)
        elif pred == "negative":
            result["negatives"].append(item)
        else:
            result["neutral"].append(item)
    return result


# ---------- Visualisasi ----------
def generate_wordcloud(data):
    text = " ".join(data["full_text"].tolist())
    wc = WordCloud(
        stopwords=set(get_stopwords()),
        background_color="black",
        max_words=500,
        width=800,
        height=400,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def plot_top_words(data, n=12, exclude_keyword=False):
    stop = set(get_stopwords())
    if exclude_keyword:
        stop.update(["rocky", "gerung"])

    words = [w for w in " ".join(data["full_text"].tolist()).split() if w not in stop]
    labels, counts = zip(*Counter(words).most_common(n))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=plt.cm.Paired(range(len(labels))))
    ax.set_xlabel("Kata")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Kata yang sering muncul")
    plt.setp(ax.get_xticklabels(), rotation=45)

    for bar, num in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 1, str(num), ha="center", fontsize=10)
    st.pyplot(fig)

def plot_sentiment_distribution(counts):
    counts = counts.reindex(SENTIMENT_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values, color=[SENTIMENT_COLORS[s] for s in counts.index])
    ax.set_title("Distribusi Sentimen")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah")
    ax.grid(axis="y", alpha=0.3)
    for bar, num in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 2, int(num), ha="center")
    st.pyplot(fig)


# ---------- KMeans ----------
@st.cache_data
def run_kmeans(texts, k):
    vectorizer = TfidfVectorizer(stop_words=get_stopwords(), max_features=5000)
    X = vectorizer.fit_transform(list(texts))

    model = KMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    terms = vectorizer.get_feature_names_out()
    order = model.cluster_centers_.argsort()[:, ::-1]
    cluster_terms = [[terms[i] for i in order[c, :10]] for c in range(k)]

    pca = PCA(n_components=2, random_state=0)
    points = pca.fit_transform(X.toarray())
    centers = pca.transform(model.cluster_centers_)
    return labels, score, cluster_terms, points, centers


# ---------- Halaman ----------
def page_home(pipeline):
    st.subheader("Home")
    st.write("Masukkan teks (tweet) untuk diprediksi sentimennya.")

    with st.form("nlpForm"):
        raw_text = st.text_area("Enter Text Here")
        submit = st.form_submit_button("Analyze")

    if submit:
        clean_text = cleansing(raw_text)
        if not clean_text:
            st.warning("Teks kosong setelah dibersihkan. Masukkan kalimat yang berisi kata.")
            return

        col1, col2 = st.columns(2)
        sentiment = pipeline.predict([clean_text])[0]

        with col1:
            st.info("Results")
            st.write(f"Teks setelah cleaning: {clean_text}")
            st.write(f"Predicted Sentiment: {sentiment}")
            if sentiment == "positive":
                st.markdown("## Positive 😃")
            elif sentiment == "negative":
                st.markdown("## Negative 😡")
            else:
                st.markdown("## Neutral 😐")

            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba([clean_text])[0]
                st.write("Probabilitas:")
                st.dataframe(
                    pd.DataFrame({"sentiment": pipeline.classes_, "probabilitas": proba.round(4)}),
                    hide_index=True,
                )

        with col2:
            st.info("Token Sentiment")
            st.json(analyze_token_sentiment(clean_text, pipeline))


def page_dataset():
    st.subheader("Dataset & Sentiment Analysis")
    data = load_data()

    st.write("### Dataset")
    st.write(f"Jumlah data: {len(data)} tweet")
    st.dataframe(data, use_container_width=True)

    st.write("### Distribusi Sentimen")
    counts = data["sentiment"].value_counts()
    plot_sentiment_distribution(counts)

    st.write("### Jumlah Sentimen")
    st.dataframe(counts.rename("jumlah"))

    st.write("### Word Cloud")
    generate_wordcloud(data)

    st.write("### Most Frequent Words")
    plot_top_words(data)


def page_kmeans():
    st.subheader("KMeans Clustering Analysis")
    data = load_data()

    k = st.slider("Jumlah cluster (k)", min_value=2, max_value=12, value=8)
    labels, score, cluster_terms, points, centers = run_kmeans(tuple(data["full_text"]), k)

    st.write("### Cluster Terms")
    for i, words in enumerate(cluster_terms):
        st.text(f"Cluster {i}: {', '.join(words)}")

    st.write(f"### Silhouette Score: {score:.4f}")
    st.caption("Nilai mendekati 0 berarti cluster saling tumpang tindih; mendekati 1 berarti terpisah jelas.")

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, c="red", linewidths=2)

    handles, _ = scatter.legend_elements()
    centroid_handle = plt.Line2D([], [], marker="x", color="red", linestyle="None", markersize=10)
    ax.legend(
        handles + [centroid_handle],
        [f"Cluster {i}" for i in sorted(set(labels))] + ["Centroid"],
        title="Keterangan",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.set_title("Visualisasi Clustering KMeans (PCA 2 Dimensi)")
    ax.set_xlabel("PCA Komponen 1")
    ax.set_ylabel("PCA Komponen 2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def page_about():
    st.subheader("About")
    st.write(
        "Aplikasi ini melakukan analisis sentimen terhadap tweet berkata kunci \"Rocky Gerung\" "
        "hasil crawling dari X (Twitter). Alur: crawling, cleaning, clustering KMeans, "
        "pelabelan sentimen, dan klasifikasi dengan TF-IDF + Logistic Regression."
    )
    st.write("Dibuat oleh: Muhammad Imam Sudais (24051130020)")


def main():
    st.title("Sentiment Analysis NLP App")
    st.caption("Streamlit Projects")

    menu = ["Home", "Dataset & Analysis", "KMeans Clustering", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        page_home(load_pipeline())
    elif choice == "Dataset & Analysis":
        page_dataset()
    elif choice == "KMeans Clustering":
        page_kmeans()
    else:
        page_about()


if __name__ == "__main__":
    main()