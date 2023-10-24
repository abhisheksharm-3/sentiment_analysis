import tkinter as tk
from tkinter import ttk
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading
import matplotlib.pyplot as plt
import html
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import credentials

lemmatizer = WordNetLemmatizer()

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

analyzer = SentimentIntensityAnalyzer()


def clean_and_tokenize(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words
    ]
    return " ".join(tokens)


def analyze_sentiment():
    query = query_entry.get()
    num_tweets = int(num_tweets_entry.get())
    analyze_button.config(state="disabled")
    progress_bar.start()

    def analysis_task():
        intermediate_label.config(text="Getting data...")
        url = "https://twitter-scraper2.p.rapidapi.com/search"
        querystring = {"searchTerms": query, "maxTweets": str(num_tweets)}
        headers = {
            "X-RapidAPI-Key": credentials.X_KEY,
            "X-RapidAPI-Host": "twitter-scraper2.p.rapidapi.com",
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            intermediate_label.config(text="Data received. Cleaning data...")
            sentiment_labels = []
            sentiment_text = []

            for tweet_data in data["data"]:
                tweet_text = tweet_data["tweet"]["full_text"]
                cleaned_text = clean_and_tokenize(tweet_text)

                sentiment = analyzer.polarity_scores(cleaned_text)
                sentiment_label = (
                    "Positive"
                    if sentiment["compound"] > 0
                    else "Negative"
                    if sentiment["compound"] < 0
                    else "Neutral"
                )

                sentiment_labels.append(sentiment_label)
                sentiment_text.append(
                    f"Tweet: {tweet_text}\nSentiment: {sentiment_label}"
                )

            intermediate_label.config(text="Analysis completed.")
            display_results(sentiment_labels, sentiment_text)
        else:
            intermediate_label.config(text="No tweets found for the given query.")
            display_results([], ["No tweets found for the given query"])

        analyze_button.config(state="normal")
        progress_bar.stop()

    analysis_thread = threading.Thread(target=analysis_task)
    analysis_thread.start()


def clean_tweet_text(text):
    cleaned_text = html.unescape(text)
    return cleaned_text


def display_results(sentiment_labels, sentiment_text):
    result_text.config(state="normal")
    result_text.delete(1.0, tk.END)

    # Count the number of tweets in each category
    positive_count = sentiment_labels.count("Positive")
    negative_count = sentiment_labels.count("Negative")
    neutral_count = sentiment_labels.count("Neutral")

    # Calculate percentages
    total_tweets = len(sentiment_labels)
    positive_percentage = (positive_count / total_tweets) * 100
    negative_percentage = (negative_count / total_tweets) * 100
    neutral_percentage = (neutral_count / total_tweets) * 100

    # Display overall sentiment percentages
    result_text.insert(tk.END, "Overall Sentiment:\n")
    result_text.insert(tk.END, f"Positive: {positive_percentage:.2f}%\n")
    result_text.insert(tk.END, f"Negative: {negative_percentage:.2f}%\n")
    result_text.insert(tk.END, f"Neutral: {neutral_percentage:.2f}%\n\n")

    # Plot sentiment analysis results
    plot_sentiment_analysis(sentiment_labels, sentiment_text)

    # Display sentiment of each tweet
    result_text.insert(tk.END, "Sentiment of each tweet:\n")
    for i, text in enumerate(sentiment_text):
        cleaned_text = clean_tweet_text(text)
        result_text.insert(tk.END, f"Tweet {i + 1}:\n{cleaned_text}\n\n")

    result_text.config(state="disabled")


def plot_sentiment_analysis(sentiment_labels, sentiment_text):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Assign colors to sentiment labels
    color_mapping = {"Positive": "g", "Negative": "r", "Neutral": "b"}
    colors = [color_mapping[label] for label in sentiment_labels]

    ax.bar(sentiment_labels, [1] * len(sentiment_labels), color=colors)
    ax.set_title("Sentiment Analysis")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    plt.tight_layout()

    for i, txt in enumerate(sentiment_text):
        ax.text(sentiment_labels[i], 1.1, txt, fontsize=10, ha="center", va="center")


app = tk.Tk()
app.title("Sentiment Analysis App")

frame = ttk.Frame(app)
frame.grid(column=0, row=0, padx=10, pady=10)

query_label = ttk.Label(frame, text="Query:")
query_label.grid(column=0, row=0, sticky="W")

query_entry = ttk.Entry(frame)
query_entry.grid(column=1, row=0, padx=10, sticky="W")

num_tweets_label = ttk.Label(frame, text="Number of Tweets:")
num_tweets_label.grid(column=0, row=1, sticky="W")

num_tweets_entry = ttk.Entry(frame)
num_tweets_entry.grid(column=1, row=1, padx=10, sticky="W")

analyze_button = ttk.Button(frame, text="Analyze", command=analyze_sentiment)
analyze_button.grid(column=0, row=2, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame, mode="indeterminate")
progress_bar.grid(column=0, row=3, columnspan=2)

intermediate_label = ttk.Label(frame, text="")
intermediate_label.grid(column=0, row=4, columnspan=2)

result_text = tk.Text(frame, wrap=tk.WORD, width=100, height=40)
result_text.grid(column=0, row=5, columnspan=2)
result_text.config(state="disabled")

app.mainloop()
