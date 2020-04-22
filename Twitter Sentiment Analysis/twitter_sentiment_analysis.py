import nltk
from nltk.corpus import twitter_samples

# Have a look at files present in twitter_samples
print(twitter_samples.fileids())
print()

# Extract the tweets from each file to respective variables
negative_tweets = twitter_samples.strings('negative_tweets.json')
positive_tweets = twitter_samples.strings('positive_tweets.json')
just_tweets = twitter_samples.strings('tweets.20150430-223406.json')

# Before tokenization, let's download an additional resource, punktThe punkt module is a pre-trained model
# that helps you tokenize words and sentences. For instance, this model knows that a name may contain a period
# (like “S. Daityari”) and the presence of this period in a sentence does not necessarily end it.
nltk.download('punkt')

# Now, lets tokenize 'positive_tweets.json' using the tokenized() method of twitter data
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

