"""
NOTE : If you are viewing this .py file first in this repository, I would recommend you to go through 'twitter_sentiment_analysis.py' file
as the code there is well commented with all explanations and it would be easy to understand the few basics needed for this project.

This file is just a structured and much neater version of the same file.
"""

# Importing the relevant libraries
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from nltk import FreqDist,classify, NaiveBayesClassifier
import random
from nltk.tokenize import word_tokenize


# function to remove noise(hyperlinks, @, punctuations) and normalize (stemming and lemmatize) tweets
def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# function to yield individual tokens into a list
def get_all_words(cleaned_token_list):
    for tokens in cleaned_token_list:
        for token in tokens:
            yield token


# function to convert tweet tokens to dictionaries with token as key and True as value
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == '__main__':
    # Extract the tweets from each file to respective variables
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    just_tweets = twitter_samples.strings('tweets.20150430-223406.json')

    # Now, lets tokenize .json files using the tokenized() method of twitter data
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    # noise removal from out tweets data using nltk stopwords
    stop_words = stopwords.words('english')
    positive_cleaned_token_list = []
    negative_cleaned_token_list = []
    for tokens in positive_tweet_tokens:
        positive_cleaned_token_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_tweet_tokens:
        negative_cleaned_token_list.append(remove_noise(tokens, stop_words))

    # Computing the word density/frequency
    all_pos_words = get_all_words(positive_cleaned_token_list)
    freq_dist_pos = FreqDist(all_pos_words)
    print(f"Most common words in the tweets data are : {freq_dist_pos.most_common(10)}")
    print()

    # Prepare the dataset for model training by providing each tweet with a sentiment, shuffling the combined
    # dataset and splitting the dataset into 70:30 train-test data
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_token_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_token_list)
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    dataset = positive_dataset + negative_dataset
    #shuffle the data
    random.shuffle(dataset)
    # Now split the data
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    # Train the model using nltk NaiveBayesClassifier
    classifier = NaiveBayesClassifier.train(train_data)

    # Accuracy using model classified data and test data.
    print(f"Accuracy from our model is : {classify.accuracy(classifier, test_data) * 100} %")
    print()
    print(classifier.show_most_informative_features(10))
    print()

    ###### ANALYSING THE SENTIMENT OF CUSTOM(USER INPUT) TWEETS #####
    #custom_tweet = "I think Machine Learning is kind of cool..."
    custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    #custom_tweet = "Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies"
    cleaned_custom_tokens = remove_noise(word_tokenize(custom_tweet), stopwords.words('english'))
    result = classifier.classify(dict([token, True] for token in cleaned_custom_tokens))
    print(f"Sentiment of user input tweet is: {result}")