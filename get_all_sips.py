from nltk.tokenize.casual import TweetTokenizer as Tok
from tweat.stream import MySQLInterface as sql
import sys, random
db=sql('localhost', 'tweat', 'tweat', 'gun_sample')
T=Tok(preserve_case=False, strip_handles=True)
corpus1=db.query('SELECT TEXT FROM CONTROL_TWEETS')
def get_sips(corpus, wcu=None, kwfilter=None, p=1.0, get_tweet=lambda x: x[0], DEBUG_PRINT=False):
   it=0
   wc={0:0}
   for line in corpus:
        if DEBUG_PRINT:
                sys.stderr.write(tweet+'\n')
                sys.stderr.flush()
        try:
                tweet=get_tweet(line)
        except IndexError:
                continue
        if random.random()>p:
                continue
        toks=T.tokenize(tweet)
        toks+=[toks[i]+' '+toks[i+1] for i in range(len(toks)-1)]
        for word in toks:
                if kwfilter is None or kwfilter in toks:
                        if word not in wc:
                                wc[word]=0
                        wc[word]+=1
                        wc[0]+=1
        if it%1000==0:
                sys.stderr.write(str(it)+' '+str(len(wc))+'\n')
                sys.stderr.flush()
        it+=1
   return wc
words_all=get_sips(corpus1)
words_kwd=get_sips(open('gun_corpus2.uniq','r'),get_tweet=lambda x: x.split(',')[4],p=1.0)
factor=1.0
def any(l,cond):
        for x in l:
                if cond(x):
                        return True
        return False
for wl in open('dict.txt','r'):
     w=' '.join(wl.split()[:-1])
     l=int(wl.split()[-1])
     if any(w.split(), lambda x: x not in words_kwd):
        continue
     if any(w.split(), lambda x: x not in words_all):
        continue
     f1=1
     f0=1
     for w0 in w.split():
             f1*=(words_kwd[w0])/(words_kwd[0])
             f0*=words_all[w0]/words_all[0]
     if (l==1 and f0<f1) or f0<f1/1000:
             print(w,f1/f0,f1,f0,l)

