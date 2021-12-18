PATTERN =  r"(doge|shib)"
TRANSFORMER_MODEL = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'
SENTIMENT_MAPPING = {
        'Positive' : 1,
        'Neutral' : 0,
        'Negative' : -1
        }
