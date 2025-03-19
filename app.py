# app.py
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import os
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
print("Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=False)
    nltk.download('punkt_tab', quiet=False)
    print("NLTK resources downloaded successfully.")
except Exception as e:
    raise ValueError(f"Failed to download NLTK resources: {str(e)}")

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("punkt_tab resource found.")
except LookupError:
    raise ValueError("punkt_tab resource not found after download. Please check NLTK installation.")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define functions for processing and analysis
def analyze_text(user_input):
    try:
        sentence_df, doc_df, error = process_corpus(user_input.splitlines())
        if error:
            return pd.DataFrame(), pd.DataFrame(), error, None
        return sentence_df, doc_df, None, doc_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Unexpected error: {str(e)}", None

def analyze_file(file, doc_id_col=None, text_col=None):
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.txt':
            with open(file.name, 'r', encoding='utf-8') as f:
                corpus = f.read().splitlines()
            sentence_df, doc_df, error = process_corpus(corpus)
            if error:
                return pd.DataFrame(), pd.DataFrame(), error, None
            return sentence_df, doc_df, None, doc_df
        elif file_ext == '.csv':
            df = pd.read_csv(file.name)
            if doc_id_col not in df.columns or text_col not in df.columns:
                return pd.DataFrame(), pd.DataFrame(), f"Selected columns '{doc_id_col}' or '{text_col}' not found in CSV.", None
            corpus = df[text_col].astype(str).tolist()
            doc_ids = df[doc_id_col].tolist()
            sentence_df, doc_df, error = process_corpus_with_ids(corpus, doc_ids)
            if error:
                return pd.DataFrame(), pd.DataFrame(), error, None
            return sentence_df, doc_df, None, doc_df
        else:
            return pd.DataFrame(), pd.DataFrame(), "Unsupported file type. Please upload a .txt or .csv file.", None
    except UnicodeDecodeError:
        try:
            with open(file.name, 'r', encoding='latin-1') as f:
                corpus = f.read().splitlines()
            sentence_df, doc_df, error = process_corpus(corpus)
            if error:
                return pd.DataFrame(), pd.DataFrame(), error, None
            return sentence_df, doc_df, None, doc_df
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame(), f"Error processing file: {str(e)}", None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Error processing file: {str(e)}", None

def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def process_corpus(corpus):
    results = []
    doc_scores = []
    for doc_idx, doc in enumerate(corpus, start=1):
        if doc.strip():
            try:
                sentences = sent_tokenize(doc.strip())
                doc_compound_score = 0
                for sent_idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():
                        senti_scores = analyzer.polarity_scores(sentence)
                        senti_scores_rounded = {k: round(v, 3) for k, v in senti_scores.items()}
                        results.append({
                            'doc_ID': doc_idx,
                            'sent_ID': sent_idx,
                            'sentence': sentence,
                            'compound': senti_scores_rounded['compound'],
                            'neg': senti_scores_rounded['neg'],
                            'neu': senti_scores_rounded['neu'],
                            'pos': senti_scores_rounded['pos'],
                            'sentiment_label': get_sentiment_label(senti_scores_rounded['compound'])
                        })
                        doc_compound_score += senti_scores_rounded['compound']
                doc_scores.append({
                    'doc_ID': doc_idx,
                    'doc_senti_score': round(doc_compound_score, 3),
                    'doc_sentiment_label': get_sentiment_label(round(doc_compound_score, 3))
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error processing document {doc_idx}: {str(e)}"
    sentence_df = pd.DataFrame(results)
    doc_df = pd.DataFrame(doc_scores)
    return sentence_df, doc_df, None

def process_corpus_with_ids(corpus, doc_ids):
    results = []
    doc_scores = []
    for doc_id, doc in zip(doc_ids, corpus):
        if doc.strip():
            try:
                sentences = sent_tokenize(doc.strip())
                doc_compound_score = 0
                for sent_idx, sentence in enumerate(sentences, start=1):
                    if sentence.strip():
                        senti_scores = analyzer.polarity_scores(sentence)
                        senti_scores_rounded = {k: round(v, 3) for k, v in senti_scores.items()}
                        results.append({
                            'doc_ID': doc_id,
                            'sent_ID': sent_idx,
                            'sentence': sentence,
                            'compound': senti_scores_rounded['compound'],
                            'neg': senti_scores_rounded['neg'],
                            'neu': senti_scores_rounded['neu'],
                            'pos': senti_scores_rounded['pos'],
                            'sentiment_label': get_sentiment_label(senti_scores_rounded['compound'])
                        })
                        doc_compound_score += senti_scores_rounded['compound']
                doc_scores.append({
                    'doc_ID': doc_id,
                    'doc_senti_score': round(doc_compound_score, 3),
                    'doc_sentiment_label': get_sentiment_label(round(doc_compound_score, 3))
                })
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), f"Error processing document {doc_id}: {str(e)}"
    sentence_df = pd.DataFrame(results)
    doc_df = pd.DataFrame(doc_scores)
    return sentence_df, doc_df, None

def generate_interactive_plot(doc_df):
    if doc_df.empty:
        return None
    fig = px.line(
        doc_df,
        x='doc_ID',
        y='doc_senti_score',
        title="Document-Level Sentiment Scores",
        labels={'doc_ID': 'Document ID', 'doc_senti_score': 'Sentiment Score'},
        hover_data={'doc_ID': True, 'doc_senti_score': True},
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7, annotation_text="Neutral")
    fig.update_layout(showlegend=False)
    return fig

def update_file_ui(file):
    try:
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(file.name)
            columns = df.columns.tolist()
            preview = df.head()
            return (
                preview,
                gr.Dropdown(choices=columns, value=columns[0], visible=True),
                gr.Dropdown(choices=columns, value=columns[1] if len(columns) > 1 else columns[0], visible=True),
                gr.Button(visible=True)
            )
        elif file_ext == '.txt':
            with open(file.name, 'r', encoding
