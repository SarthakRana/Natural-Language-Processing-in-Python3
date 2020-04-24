# STEP 1 - Importing the nltk library and downloading the twitter_samples data
import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

# Have a look at files present in twitter_samples
print(twitter_samples.fileids())

# STEP 2 - Tokenization
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

# STEP 3 - Normalization - Stemming & Lemmatization
# Words have different forms—for instance, “ran”, “runs”, and “running” are various forms of the same verb,
# “run”. Depending on the requirement of your analysis, all of these versions may need to be converted to the
# same form, “run”. Normalization in NLP is the process of converting a word to its canonical form.

# wordnet helps the script determine the base word.
nltk.download('wordnet') # For stemming
# averaged_perceptron_tagger helps determine the context of a word in a sentence.
nltk.download('averaged_perceptron_tagger') # For lemmatization

# Before using lemmatizer, we need to determine the context for each word in our text.
# This is achieved by a tagging algorithm i.e pos_tag()
#print(pos_tag(tweet_tokens[0])) # Uncomment this line to see how tagging actually looks like in a sentence.

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


#print(lemmatize_sentence(tweet_tokens[0]))
