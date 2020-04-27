############ STEP 1 - Importing the nltk library and downloading the twitter_samples data ##########

import nltk
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
from nltk import FreqDist
import random


# Have a look at files present in twitter_samples
print(twitter_samples.fileids())


############# STEP 2 - Tokenization ##############

# Extract the tweets from each file to respective variables
negative_tweets = twitter_samples.strings('negative_tweets.json')
positive_tweets = twitter_samples.strings('positive_tweets.json')
just_tweets = twitter_samples.strings('tweets.20150430-223406.json')

# Before tokenization, let's download an additional resource, punktThe punkt module is a pre-trained model
# that helps you tokenize words and sentences. For instance, this model knows that a name may contain a period
# (like “S. Daityari”) and the presence of this period in a sentence does not necessarily end it.
# nltk.download('punkt') # uncomment this line to run first time only

# Now, lets tokenize .json files using the tokenized() method of twitter data
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')


############# STEP 3 - Normalization - Stemming & Lemmatization #############

# Words have different forms—for instance, “ran”, “runs”, and “running” are various forms of the same verb,
# “run”. Depending on the requirement of your analysis, all of these versions may need to be converted to the
# same form, “run”. Normalization in NLP is the process of converting a word to its canonical form.

# wordnet helps the script determine the base word.
#nltk.download('wordnet') # uncomment this line to run first time only
# averaged_perceptron_tagger helps determine the context of a word in a sentence.
#nltk.download('averaged_perceptron_tagger') # uncomment this line to run first time only

# Before using lemmatizer, we need to determine the context for each word in our text.
# This is achieved by a tagging algorithm i.e pos_tag()
#print(pos_tag(tweet_tokens[0])) # Uncomment this line to see how tagging actually looks like in a sentence.

# def lemmatize_sentence(tokens):
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_sentence = []
#     for word, tag in pos_tag(tokens):
#         if tag.startswith('NN'):
#             pos = 'n'
#         elif tag.startswith('VB'):
#             pos = 'v'
#         else:
#             pos = 'a'
#         lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
#     return lemmatized_sentence


################# STEP 4 — Removing Noise from the Data ###############

# Noise is any part of the text that does not add meaning or information to data. For instance, the most common
# words in a language are called stop words. Some examples of stop words are “is”, “the”, and “a”. They are
# generally irrelevant when processing language, unless a specific use case warrants their inclusion.

# In this project, we will use Regex to remove :- (1)Hyperlinks (2)Twitter handles (3)Punctuation and special characters
# Below function will remove all noise from text and then normalize each token to return cleaned tokens.
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

#nltk.download('stopwords') # uncomment this line to run first time only
stop_words = stopwords.words('english')

positive_cleaned_token_list = []
negative_cleaned_token_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_token_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_token_list.append(remove_noise(tokens, stop_words))


############### STEP 5 - DETERMINING WORD DENSITY ###############
# Now, we will compute the frequency of each word in our positive tokens list
# We want all tokens inside 'positive_cleaned_token_list' in a single list.
def get_all_words(cleaned_token_list):
    for tokens in cleaned_token_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_token_list)
freq_dist_pos = FreqDist(all_pos_words)
#print(freq_dist_pos.most_common(10)) # Uncomment this line to see top 10 tokens with most frequency

#From this data, you can see that emoticon entities form some of the most common parts of positive tweets.


############### STEP 6 - PREPARING DATA FOR MODEL ###############
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_token_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_token_list)


positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

# data in dataset list is stored as tuples
dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

# Now split the data
train_data = dataset[:7000]
test_data = dataset[7000:]


################# STEP 7 — BUILDING AND TESTING THE MODEL #################
# Here we will use the NaiveBayedClassifier of nltk library to build our model

from nltk import classify, NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

print(f"Accuracy is : {classify.accuracy(classifier, test_data)*100} %")
print(classifier.show_most_informative_features(10))



# Lets analyse the sentiment of a custom_tweet
from nltk.tokenize import word_tokenize
#custom_tweet = "I think Machine Learning is kind of cool..."
custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
#custom_tweet = "Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies"
custom_tweet_token = word_tokenize(custom_tweet)
nltk.download('stopwords')
stop_words = stopwords.words('english')

cleaned_custom_tokens = remove_noise(custom_tweet_token, stop_words)
print(cleaned_custom_tokens)

result = classifier.classify(dict([token, True] for token in cleaned_custom_tokens))
print("Sentiment of custom tweet : ", result)
