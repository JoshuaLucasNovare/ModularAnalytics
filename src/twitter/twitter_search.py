import os
import tweepy
import pandas as pd
from decouple import config
from .utils import get_language, remove_enter, translate_to_eng

def tw_oauth():

    consumer_key = os.environ['CONSUMER_KEY']
    consumer_secret = os.environ['CONSUMER_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_token_secret = os.environ['ACCESS_TOKEN_SECRET']

    auth = tweepy.auth.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def tw_search(api, keywords):
    keyword = "Landbank"
    keyword = keywords

    tweets = tweepy.Cursor(api.search, q=keyword)
    tweet_list = {
        'source': [],
        'tweet': [],
        'date_posted': []
    }

    for tweet in tweets.items():
        tweet_list['source'].append("Twitter")
        tweet_list['tweet'].append(tweet.text)
        tweet_list['date_posted'].append(tweet.created_at)

    tweet_df = pd.DataFrame(tweet_list)

    # Change this to use Spark
    tweet_df['tweet'] = tweet_df['tweet'].apply(remove_enter)
    # tweet_df['orignal_language'] = tweet_df['tweet'].apply(get_language)
    # tweet_df['translated'] = tweet_df['tweet'].apply(translate_to_eng)
    # tweet_df.to_csv('tweets2.csv', index=False)
    return tweet_df

def get_tweets(keywords):
    api = tw_oauth()

    return tw_search(api, keywords)
































































