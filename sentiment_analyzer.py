import tkinter as tk
from tkinter import ttk
import requests
from textblob import TextBlob
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import credentials
import threading
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

def clean_and_tokenize(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def perform_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def get_sentiment_label(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment():
    query = query_entry.get()
    num_tweets = int(num_tweets_entry.get())
    analyze_button.config(state="disabled")
    progress_bar.start()

    def analysis_task():
        intermediate_label.config(text="Getting data...")
        app.update()  # Force the GUI to update

        url = "https://twitter-scraper2.p.rapidapi.com/search"
        querystring = {"searchTerms": query, "maxTweets": str(num_tweets)}
        headers = {
            "X-RapidAPI-Key": credentials.X_KEY,
            "X-RapidAPI-Host": "twitter-scraper2.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()

        if 'data' in data and len(data['data']) > 0:
            intermediate_label.config(text="Data received. Cleaning data...")
            app.update()  # Force the GUI to update

            sentiment_labels = []
            sentiment_text = []
            stemmed_tweets = []

            for tweet_data in data['data']:
                tweet_text = tweet_data['tweet']['full_text']
                cleaned_text = clean_and_tokenize(tweet_text)
                analysis = TextBlob(cleaned_text)

                sentiment = get_sentiment_label(analysis.sentiment.polarity)
                sentiment_labels.append(sentiment)
                sentiment_text.append(f"Tweet: {tweet_text}\nSentiment: {sentiment}")

                stemmed_text = perform_stemming(cleaned_text)
                stemmed_tweets.append(stemmed_text)

            with open('stemmed_tweets.json', 'w') as json_file:
                json.dump(stemmed_tweets, json_file, indent=4)

            intermediate_label.config(text="Analysis completed.")
            app.update()  # Force the GUI to update
            display_results(sentiment_labels, sentiment_text)
        else:
            intermediate_label.config(text="No tweets found for the given query.")
            app.update()  # Force the GUI to update
            display_results([], ["No tweets found for the given query."])

        analyze_button.config(state="normal")
        progress_bar.stop()

    analysis_thread = threading.Thread(target=analysis_task)
    analysis_thread.start()


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
    result_text.insert(tk.END, f"Overall Sentiment:\n")
    result_text.insert(tk.END, f"Positive: {positive_percentage:.2f}%\n")
    result_text.insert(tk.END, f"Negative: {negative_percentage:.2f}%\n")
    result_text.insert(tk.END, f"Neutral: {neutral_percentage:.2f}%\n\n")

    # Display sentiment of each tweet
    for i, text in enumerate(sentiment_text):
        result_text.insert(tk.END, f"Tweet {i + 1}:\n{text}\n\n")
    result_text.config(state="disabled")

    # Plot sentiment analysis results
    plot_sentiment_analysis(sentiment_labels, sentiment_text)

def plot_sentiment_analysis(sentiment_labels, sentiment_text):
    # Create a bar chart for sentiment labels
    num_tweets = len(sentiment_labels)
    tweet_labels = [f"Tweet {i+1}" for i in range(num_tweets)]
    
    # Assign colors to sentiment labels (you can customize these)
    color_mapping = {"Positive": 'g', "Negative": 'r', "Neutral": 'b'}
    colors = [color_mapping[label] for label in sentiment_labels]

    plt.figure(figsize=(12, 8))
    plt.bar(tweet_labels, [1] * num_tweets, color=colors)
    plt.title("Sentiment Analysis")
    plt.xlabel("Tweets")
    plt.ylabel("Sentiment")
    
    # Display text labels
    for i, txt in enumerate(sentiment_text):
        plt.text(i, 0.5, txt, fontsize=10, ha='center', va='center')

    plt.xticks([])  # Hide x-axis labels
    plt.show()

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

result_text = tk.Text(frame, wrap=tk.WORD, width=50, height=10)
app.mainloop()