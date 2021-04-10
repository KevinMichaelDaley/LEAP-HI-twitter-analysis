import tweepy
import time

class TwitterInterface:
    def __init__(self,consumer_key, consumer_secret, access_token, access_secret):
        '''Initializes a twitter api object using the given keys, which should not be human-readable in the application'''
        self.tweepy_auth=tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.tweepy_auth.set_access_token(access_token, access_secret)
        self.tweepy_api = tweepy.API(self.tweepy_auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    def stream_tweets_by_user(self,username, callback, N=3200, rts=True):
        '''for example, stream_tweets_by_user('realdonaldtrump'). returns iterator of statuses; see tweepy api.  will throw TweepError if rate-limited'''
        for status in tweepy.Cursor(self.tweepy_api.user_timeline, username, count=N, include_rts=rts, tweet_mode='extended').items():
            callback(status)
    def build_network(self,username, callback, max_friends=1500):
        users=[]
        for user in tweepy.Cursor(self.tweepy_api.friends, screen_name=username).items():
           callback(user)
        return users
    def stream_tweets_by_topic(self,topic, N=1000, **kwargs):
        '''for example, stream_tweets_by_topic('ketchup') or stream_tweets_by_topic('march for ketchup'). returns iterator of statuses.  throws TweepError if rate-limited'''
        return tweepy.Cursor(self.tweepy_api.search, q=topic, count=N,lang="en",tweet_mode="extended", **kwargs).items()
    def get_trends_by_place(self,location_code=1):
        '''for location codes refer to the inbuilt constants or twitter API docs.  1 is the world. throws TweepError if rate-limited'''
        return [trend['name'] for trend in self.tweepy_api.trends_place(location_code)[0]['trends']]
    def stream_trending_tweets_by_place(self,location_code=1, N_per_topic=100):
        '''for location codes refer to the  builtin query method or twitter API docs.  1 is the world.  throws TweepError if rate-limited; pulls N_per_topic tweets for each trend.  Each trend is one GET request.'''
        trends=self.get_trends_by_place(location_code)
        tweets={}
        for trend in trends:
            tweets[trend]=self.stream_tweets_by_topic(trend, N)
        return tweets
    def get_location_id(self, name, kind='neighborhood'):
        '''for example, get_location_id('USA', 'country')'''
        places = self.tweepy_api.geo_search(query=name, granularity=kind)
        return places[0].id

