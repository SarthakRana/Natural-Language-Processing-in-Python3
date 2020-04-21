"""
In this tutorial, we will see the NLTK in action on twitter data. NLTK’s twitter corpus currently contains
a sample of 20,000 tweets retrieved from the Twitter Streaming API. Full tweets are stored as line-separated JSON.

The goal of our script will be to count how many adjectives and nouns appear in the positive subset of the
twitter_samples corpus
"""
# STEP 1 - Importing NLTK

# Importing Natural Language Toolkit - nltk
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents
import os

# Check if the latest version is installed. If not, do so.
print()
print(nltk.__version__)

# STEP 2 - Downloading NLTK’s Data and Tagger

# To download twitter corpus, run the following command in Command line
"python -m nltk.downloader twitter_samples"

# Next, download the part-of-speech (POS) tagger. In this tutorial, we will specifically use NLTK’s
# averaged_perceptron_tagger. Run the following command in Command Line
"python -m nltk.downloader averaged_perceptron_tagger"

# To check how many JSON files
print()
print(twitter_samples.fileids())

# To check how data looks like in these JSON files
# WARNING : Running this will return a lot of data. Editor may crash...
print()
path = os.getcwd()
file = "nltk_data\\corpora\\twitter_samples\\negative_tweets.json"
# with open(os.path.join(path, file)) as f:
#     print(f.read())

# STEP 3 - Tokenizing Sentences
# Tokenization is the act of breaking up a sequence of strings into pieces such as
# words, keywords, phrases, symbols and other elements, which are called tokens.

tweets = twitter_samples.strings('positive_tweets.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

# STEP 4 — Tagging Sentences
# Abreviation for Ajective = JJ
# Abreviation for Singular Nouns = NN
# Abreviation for Plural Nouns = NNS
tweets_tagged = pos_tag_sents(tweet_tokens)

# STEP 5 — Counting POS Tags
# Since our objective was to find how many adjectives and nouns appear in the positive subset of the
# twitter_samples corpus, let us count them.
JJ_count = 0
NN_count = 0
for tweet in tweets_tagged:
    for tagged_token in tweet:
        if tagged_token[1] == 'JJ':
            JJ_count+=1
        elif tagged_token[1] == 'NN':
            NN_count+=1
print(f"Total number of Adjectives : {JJ_count}")
print(f"Total number of Nouns : {NN_count}")

# Now, as a further task you can use these adjectives and nouns in other sentiment analysis project works.
# Similarly, you can find out a list of adjectives and nouns in negative JSON file and use in other NLP related projects.
