import json
import argparse
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer

def read_messages_from_file(file_name):
    with open(f"{file_name}") as file:
        print("Reading Messages")
        text = file.read()
        return json.loads(text)["messages"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()

    messages = read_messages_from_file(args.input_file)
    print(f"{len(messages)} Messages found!")

    crypto_sentiment_analyzer = CryptoSentimentAnalyzer(messages)

    crypto_sentiment_analyzer.analyze_sentiment()
    crypto_sentiment_analyzer.plot_summary()
    print("Done!")


if __name__ == '__main__':
    main()
