import re
import pandas as pd
import os
import plotly.graph_objs as go
from langdetect import detect
from transformers import pipeline
from tqdm import tqdm
from constant import PATTERN, SENTIMENT_MAPPING, TRANSFORMER_MODEL

tqdm.pandas()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CryptoSentimentAnalyzer():
    def __init__(self, messages) -> None:
        self.message_dataframe = pd.DataFrame(messages)
        self.doge_shiba_dataframe = self._pre_process_messages()
        print(f"{len(self.doge_shiba_dataframe)} 'SHIB' or 'DOGE' messages found!")

    def _pre_process_messages(self) -> None:

        print("Preprocessing messages")
        self.message_dataframe['timestamp'] = pd.to_datetime(
            self.message_dataframe['date'])
        self.message_dataframe['date'] = self.message_dataframe['timestamp'].dt.date

        print("Filtering 'DOGE' and 'SHIB' english language messages")
        self.message_dataframe['normalized_message'] = self.message_dataframe['text'].apply(
            CryptoSentimentAnalyzer.normalize_message)
        pattern = re.compile(PATTERN, re.IGNORECASE)
        self.message_dataframe['doge_shiba_filter'] = self.message_dataframe['normalized_message'].progress_apply(
            CryptoSentimentAnalyzer.filter_doge_shiba, args=[pattern])
        return self.message_dataframe[self.message_dataframe['doge_shiba_filter']].copy(
        )

    def analyze_sentiment(self):
        print(f"Dowloading/Loading {TRANSFORMER_MODEL} Transformer model")
        sentiment_analysis = pipeline(
            "sentiment-analysis", model=TRANSFORMER_MODEL)

        print(
            f"Predicting Sentiments of {len(self.doge_shiba_dataframe)} messages")
        self.doge_shiba_dataframe['senitment_result'] = \
            self.doge_shiba_dataframe['normalized_message'].progress_apply(
                sentiment_analysis)
        self.doge_shiba_dataframe['sentiment'] = self.doge_shiba_dataframe['senitment_result'].apply(
            lambda x: x[0]['label'])
        self.doge_shiba_dataframe['score'] = self.doge_shiba_dataframe['senitment_result'].apply(
            lambda x: x[0]['score'])
        self.doge_shiba_dataframe['sentiment_value'] = self.doge_shiba_dataframe['sentiment'].apply(
            lambda x: SENTIMENT_MAPPING[x])
        return self.doge_shiba_dataframe

    def plot_summary(self):
        print("Graphing Average Sentiment by date")
        sentiment_by_date = self.doge_shiba_dataframe.groupby(['date']).agg({
            'sentiment_value': 'mean',
            'normalized_message' : 'count'
        }).reset_index().rename(columns={
            'sentiment_value' : 'average_sentiment_value',
            'normalized_message' : 'total_messages'
        })

        trace1 = go.Bar(x=sentiment_by_date['date'],
                        y=sentiment_by_date['total_messages'],
                        name='Number of Messages',
                        yaxis='y1')

        trace2 = go.Scatter(x=sentiment_by_date['date'],
                            y=sentiment_by_date['average_sentiment_value'],
                            name='Avg Sentiment',
                            mode='lines+markers',
                            yaxis='y2')

        data = [trace1, trace2]

        layout = go.Layout(title='Number of Messages and Average Sentiment by Day',
                           xaxis=dict(title='Days'),
                           yaxis=dict(title='Number of Messages'),
                           yaxis2=dict(title='Avg Sentiment',
                                       overlaying='y',
                                       side='right',
                                       range=[-1, 1]))
        fig = go.Figure(data=data, layout=layout)
        fig.show()

    @staticmethod
    def normalize_message(message):
        if isinstance(message, list):
            return "".join([item for item in message if isinstance(item, str)])
        return message

    @staticmethod
    def filter_doge_shiba(message, pattern):
        return pattern.search(message) is not None and detect(message) == 'en'
