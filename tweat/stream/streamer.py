import sys,time,os
from . import twitter_hooks, mysql_hooks
from tweepy import RateLimitError,Stream,StreamListener
import signal, random
class TweatStreamListener(StreamListener):
    def __init__(self,twitter,db,name,filters):
        self.twitter=twitter
        self.db=db
        self.name=name
        self.api=self.twitter.tweepy_api
        self.tweets=0
        self.filters=filters
    def on_status(self,status):
        print(status.user.screen_name,": ",status.text)
        if not hasattr(status, 'extended_tweet'):
            full_text=status.text
        else:
            full_text=status.extended_tweet['full_text']
        if hasattr(status, 'retweeted_status'):
            rtstatus=status.retweeted_status

            if not hasattr(rtstatus, 'extended_tweet'):
                full_text+=rtstatus.text
            else:
                full_text+=rtstatus.extended_tweet['full_text']
        contains=False
        for f in self.filters:
            if f.lower() in full_text.lower():
                contains=True
        if contains or len(self.filters)==0:
            self.db.add_tweet(status, self.name)
        else:
            self.db.add_control_tweet(status, self.name)
    def on_error(self, status_code):
        print('err:'+repr(status_code))
        sys.exit(-1)
        return False
    def on_delete(self, status_id, user_id):
        pass
    def on_limit(self, track):
        time.sleep(50)
        return False
    def on_timeout(self):
        time.sleep(10)
        return False
    def on_exception(self, exception):
        sys.exit(-1)
        return False
class TweetStreamer:
    def __init__(self,twitter,db):
        self.db=db
        self.twitter=twitter
    def build_network(self, name, name2):
            print('\t '+name[len('u:'):])
            self.db.add_user_friends(name[len('u:'):], name2[len('u:'):])
            res=self.twitter.tweepy_api.show_friendship(source_screen_name=name[len('u:'):], target_screen_name=name[len('u:'):])
            if res[1].followed_by:
                self.db.add_user_friends(self,[name[len('u:'):]], [name2[len('u:'):]])

            if res[0].followed_by:
                self.db.add_user_friends(self,[name2[len('u:'):]], [name[len('u:'):]])
    def stream(self,name,filters=[]):
        def callback(status):
            contains=False
            for f in filters:
                if f.lower() in status.full_text.lower():
                    contains=True
            if not contains and len(filters)>0:
                self.db.add_control_tweet(status, name)
            else:
                self.db.add_tweet(status, name)

        if name[0:len('u:')]!='u:':
            myStream = tweepy.Stream(auth = self.twitter.tweepy_api.auth, listener=TweatStreamListener(self.twitter,self.db,name,filters), tweet_mode='extended', timeout=300)
            try:
                myStream.sample()
            except:
                return

        else:
            self.twitter.stream_tweets_by_user(name[len('u:'):], callback)
def authorize_twitter():
    def get_keys_dir():
        kkey = 'TWEAT'
        if kkey not in dict(os.environ):
            raise KeyError("Missing env var $TWEAT (path to auth keys).")
        kdir = os.environ[kkey]
        return kdir
    kfiles = dict(ckey='.ckey', csecret='.csecret',
                  atoken='.atoken', asecret='.asecret')
    kdict = dict()
    kdir = get_keys_dir()
    for kk, kf in kfiles.items():
        with open(os.path.join(kdir, kf), 'r+') as kfr:
            kdict.update({kk: kfr.readline().rstrip()})
    twitter_iface=twitter_hooks.TwitterInterface(kdict['ckey'], kdict['csecret'],
                                                 kdict['atoken'], kdict['asecret'])
    return twitter_iface

def stream_by_topic(*args, limit=-1):
    parsed = args[0]
    running=True
    gfilters=[]
    if 'gun' in parsed['db']:
        gfilters=["gun violence", "gun rights", "second amendment", "assault weapons", " #2A ", "2nd amend", "guncontrol", "gun control", "mass shooting", "gun owner"]
    elif 'covid' in parsed['db']:
        gfilters=["covid", "coronavirus", "virus", "rona","covid19", "quarantine", "lockdown", "social distancing"]
    elif 'control' in parsed['db']:
       us_english=[]
       for line in open('google-10000-english-usa.txt'):
              if len(line.strip())>2:
                     us_english.append(line.strip())
       gfilters=us_english[:200]
    def quit_on_sigterm(signum,frame):
        running=False
    twitter_iface=authorize_twitter()
    db=mysql_hooks.MySQLInterface(parsed['host'],
                                  parsed['user'],
                                  parsed['password'],
                                  parsed['db'])
    streamer=TweetStreamer(twitter_iface,db)
    signal.signal(signal.SIGTERM, quit_on_sigterm)
    streamer.stream('', filters=gfilters)
import tweepy
def stream_all(*args, limit=-1):
    parsed = args[0]
    running=True
    def quit_on_sigterm(signum,frame):
        running=False
    twitter_iface=authorize_twitter()
    db=mysql_hooks.MySQLInterface(parsed['host'],
                                  parsed['user'],
                                  parsed['password'],
                                  parsed['db'])

    streamer=TweetStreamer(twitter_iface,db)
    signal.signal(signal.SIGTERM, quit_on_sigterm)
    fname=""
    if parsed['db']=='gun2':
        fname='users_gun.txt'
        gfilters=["gun violence", "gun rights", "second amendment", "assault weapons", " #2A ", "2nd amend", "bear arms", "guncontrol", "gun control", "mass shooting", "gun owner"]
    elif parsed['db']=='sports2':
        fname='users_sports.txt'
        gfilters=["falcons", "whodat", "rise up", "riseup", "saints", "football", "offense", "defense", "ref", "playoffs", "champion", "point lead","ryan", "julio", "brees", "jordan"]
    elif parsed['db']=='brexit2':
        fname='users_sports.txt'
        gfilters=["brexit", "eu"]
    elif parsed['db']=='control2':
        fname='users_control.txt'
        gfilters=[]
    for user in open(fname):
        print(user)
        for user2 in open(fname):
            if user==user2:
                break
            try:
                streamer.build_network('u:'+user, 'u:'+user2)
            except tweepy.TweepError as ex:
                    if ex.reason == "Not authorized.":
                        pass
        print(twitter_iface.tweepy_api.rate_limit_status())
    for t in range(10):
        for user in open(fname):
            try:
                streamer.stream('u:'+user.strip(),filters=gfilters)
                print(user)
            except tweepy.TweepError as ex:
                if ex.reason == "Not authorized.":
                    pass
            print(twitter_iface.tweepy_api.rate_limit_status())
def query_locations(*args):
    db=mysql_hooks.MySQLInterface(*args)
    twitter_iface=authorize_twitter()
    users=db.query('SELECT DISTINCT USER FROM TWEETS;')
    users_already=db.query('SELECT DISTINCT USER FROM USERS;')
    users=list(set(users)-set(users_already))
    names=[]
    for u in users:
        names.append(u[0])
    i=0
    for i in range(0,100*(len(users)//100+1),100):
       users2=twitter_iface.tweepy_api.lookup_users(screen_names=names[i:100+i])
       for u in users2:
           try:
               db.execute("INSERT INTO USERS VALUES (%i, \'%s\', \'%s\')"%(u.id, u.screen_name, u.location.replace('\'','"')))
           except:
               continue
       print(100+i, len(users))
       time.sleep(15)
"""

from bisect import bisect_left
def restream_some(*args):# to redownload tweets we have half of due to api changes.
    parsed = args[0]
    twitter_iface=authorize_twitter()
    db=mysql_hooks.MySQLInterface(parsed['host'],
                                  parsed['user'],
                                  parsed['password'],
                                  parsed['db'])
    id0q=db.query("SELECT ID FROM TWEETS WHERE TOPIC='_RETRY' GROUP BY ID ORDER BY ID DESC");
    if(len(id0q)>0):
        id0=int(id0q[0][0])
    else:
        id0=-1
    tweetstring = db.query("SELECT ID FROM TWEETS WHERE ID>%i GROUP BY ID ORDER BY ID"%id0);
    tweet_IDs=[int(x[0]) for x in tweetstring]
    print(tweet_IDs)
    sys.exit(1)
    tweet_IDs=tweet_IDs[bisect_left(tweet_IDs,id0):]
    tweet_count = len(tweet_IDs)
    for i in range(0,(tweet_count // 100) + 1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            try:
                try:
                    tweets=twitter_iface.tweepy_api.statuses_lookup(id_=tweet_IDs[i * 100:end_loc], tweet_mode='extended')
                    for tw in tweets:
                        db.add_tweet(tw,'_RETRY');
                        print(tw.full_text);
                except tweepy.TweepError:
                    continue
            except RateLimitError:
                time.sleep(5.05)
                continue

if __name__=='__main__':
    restream_some(parser.parse_args())"""
