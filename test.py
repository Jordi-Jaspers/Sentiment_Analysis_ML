from textblob import TextBlob

# Returns the sentiment of text
# By returning a value between -1.0 and 1.0
def getSentiment(messages):
    message_sentiment = []
    for message in messages:
        obj = TextBlob(message)
        polarity = obj.sentiment.polarity

        if polarity == 0:
            sentiment = 2
        elif polarity > 0:
            sentiment = 4
        else:
            sentiment = 0

        message_sentiment.append(sentiment)

    return message_sentiment