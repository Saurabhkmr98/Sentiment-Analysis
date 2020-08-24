import nltk
import pandas as pd
import re, string, random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

df = pd.read_excel(r'C:\Users\Asus\Desktop\Intellisense\Online_HRD_30052020.xlsx')
pd.set_option('display.max_columns', None)
df = df[df.Language == 'English']

posit = df[df.Sentiment == 'Positive']
positive_headlines = list(posit.Headline)
negat = df[df.Sentiment == 'Negative']
negative_headlines = list(negat.Headline)
neut = df[df.Sentiment == 'Neutral']
neutral_headlines = list(negat.Headline)

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
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

def generate_token(sentiment_list):
    list_of_tokens = []
    for element in sentiment_list:
        list_of_tokens.append(word_tokenize(element))
    return list_of_tokens

def english_token(list_of_list):
    for element in list_of_list:
        for i,ele in enumerate(element):
            src = t.detect(ele).lang
            dest = 'en'
            if src != 'en':
                element[i] = t.translate(sample, src = src, dest = 'en').text
    return list_of_list

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        return dict([token, True] for token in tweet_tokens)

if __name__ == "__main__":

    positive_tweets = positive_headlines
    print(positive_tweets)
    negative_tweets = negative_headlines
    print(negative_tweets)
    neutral_tweets = neutral_headlines
    print(neutral_tweets)

    stop_words = stopwords.words('english')

    positive_tweet_tokens = generate_token(positive_tweets)
    print(positive_tweet_tokens) # a list of list of tokens 
    negative_tweet_tokens = generate_token(negative_tweets)
    print(negative_tweet_tokens)
    neutral_tweet_tokens = generate_token(neutral_tweets)
    print(neutral_tweet_tokens)

    positive_cleaned_tokens_list = []
    neutral_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    for tokens in neutral_tweet_tokens:
        neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
        
    print(positive_cleaned_tokens_list)
    print(negative_cleaned_tokens_list)
    print(neutral_cleaned_tokens_list)


    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    
    print(positive_tokens_for_model)
    print(negative_tokens_for_model)
    print(neutral_tokens_for_model)
    
    positive_dataset = [(tweet_dict, "Positive")
                         for tweet_dict in positive_tokens_for_model]
    
    neutral_dataset = [(tweet_dict, "Neutral")
                         for tweet_dict in neutral_tokens_for_model]
    
    negative_dataset = [(tweet_dict, "Negative")
                         for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset + neutral_dataset
    print("****************************************************************")
    print(len(dataset))

    random.shuffle(dataset)

    train_data = dataset[:100]
    test_data = dataset[100:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))
    
    custom_tweet = "Supreme Court notice to Center and states in Hathini\'s death case"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))

