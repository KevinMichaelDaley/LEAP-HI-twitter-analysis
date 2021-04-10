import tweepy, networkx
import numpy as np
from MySQLdb._exceptions import ProgrammingError as MySQLError, OperationalError as MySQLOperationalError
from tweat.stream import MySQLInterface, TwitterInterface, TweetStreamer, stream_all, authorize_twitter
from tweat.tokenize import Tokenizer, tokenize_tweet
import datetime, re, sys, dateutil.parser, os
from tweat.classify import train_model, vectorize
from tweat.word2vec import Word2Vec
db=MySQLInterface(*sys.argv[1:5])
w2v=Word2Vec(db)
wordsids=db.query("SELECT DISTINCT ID, WORD FROM WORDOCCURRENCES")
words=[w[1].lower() for w in wordsids]
wids=[w[0] for w in wordsids]
model=train_model(db,False,w2v,wids)
wordlist_emo_neg=[re.compile(line.strip().replace('*','[A-z\']*')) for line in open('emo_neg.txt')]
wordlist_emo_pos=[re.compile(line.strip().replace('*','[A-z\']*')) for line in open('emo_pos.txt')]
Tok=Tokenizer()
last_id=-1
if os.path.exists('last_id'):
    last_id=int(open('last_id', 'r').read())
F=open(sys.argv[5], 'a+')
tweets=db.query('SELECT ID,USER,TEXT,ISODATE FROM TWEETS WHERE ID>%i GROUP BY ID,USER,TEXT,ISODATE ORDER BY ID'%last_id)
userloc=db.query('SELECT DISTINCT USER, LOCATION FROM USERS GROUP BY USER,LOCATION ORDER BY USER')
users=[u[0] for u in userloc]
locs=[u[1] for u in userloc] 
loccoords=db.query('SELECT DISTINCT LOCATION, LATC, LONGC FROM LOCATIONS')
locs2=[u[0] for u in loccoords]
latlong=[[float(u[1]),float(u[2])] for u in loccoords]
isodate0=datetime.datetime(2019,1,1,0,0,0,0)
re_mention=re.compile("\s@([\w_-]+)")
re_rt=re.compile("^(RT|rt) @(\w*)?[: ]")
for tweet in tweets:
    ID, user,txt,isodate=tweet
    if user not in users: 
        continue
    try:
        ui=users.index(user)
        if locs[ui]=='':
            C=[-1, -1]
        else:
            li=locs2.index(locs[ui])
            C=latlong[li]
        T=(dateutil.parser.parse(isodate)-isodate0).total_seconds()
        all_mentions=[]
        rt=False
        if '@' in txt:
            m=re_mention.finditer(txt)
            if m is not None:
                for u in m:
                    all_mentions.append(u.group(1))
            if 'rt' in txt:
                m2=re_rt.search(txt)
                if m2 is not None:
                    rt=True
                    rt_user=m2.group(2)
        ids=[int(wids[words.index(k.lower())]) if k.lower() in words else 0 for k in Tok.tokenize(txt)]
        sentiment=0
        if len(ids)>=8:
            sentiment=model.predict(np.array(vectorize(ids,db,w2v,wids)).reshape([1,-1]))>0
        emo_pos = sum([w.search(txt) is not None for w in wordlist_emo_neg])
        emo_neg = sum([w.search(txt) is not None for w in wordlist_emo_pos])
        for m in all_mentions: 
            ui2=users.index(m) if m in users else -1
            if ui2<0 or locs[ui2]=='':
                C2=[-1,-1]
            else:
                li2=locs2.index(locs[ui2])
                C2=latlong[li2]
            F.write("@ "+" ".join(list(map(str,[T,C[0], C[1], ui, C2[0], C2[1], ui2, emo_pos, emo_neg, sentiment, user, m])))+"\n")
            F.flush()
        if rt:
            ui3=users.index(rt_user) if rt_user in users else -1
            if ui3<0 or locs[ui3]=='':
                C2=[-1,-1]
            else:
                li3=locs2.index(locs[ui3])
                C3=latlong[li3]
            F.write("rt@ "+" ".join(list(map(str,[T, C[0], C[1], ui, C3[0], C3[1], ui3, emo_pos, emo_neg, sentiment, user, rt_user])))+'\n')
        F.write("twt "+" ".join(list(map(str,[T, C[0], C[1], emo_pos, emo_neg, sentiment, ID, ui, user])))+'\n')
    except:
        continue
    open('last_id', 'w+').write(str(ID))
    F.flush()





        


        
        
    
