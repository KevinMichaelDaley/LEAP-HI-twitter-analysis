from nltk.tokenize.casual import TweetTokenizer as Tok
from tweat.stream import MySQLInterface as sql
import sys
db=sql(*sys.argv[1:])
T=Tok(preserve_case=False, strip_handles=True)
words_the={}
words_gun={}
it=0
txt=db.query('SELECT TEXT FROM CONTROL_TWEETS ORDER BY RAND() LIMIT 1000000')
for tweet in txt:
        t=tweet[0]
        toks=T.tokenize(tweet[0])
        kwd='gun' if 'gun' in toks else 'the'
        for word in toks:
                if kwd=='the':
                        if word not in words_the:
                                words_the[word]=0
                        words_the[word]+=1
                if kwd=='gun':
                        if word not in words_gun:
                                words_gun[word]=0
                        words_gun[word]+=1
        if it%1000==0:
                sys.stderr.write(str(it)+' '+str(len(words_gun))+'\n')
                sys.stderr.flush()
        it+=1
for w in words_gun:
   if w in words_the:
        if words_gun[w]/len(words_gun)>words_the[w]*10.0/len(words_the):
           if words_gun[w]>10:
             print(w,words_gun[w]/len(words_gun), words_the[w]/len(words_the))
