from nltk.tokenize.casual import TweetTokenizer as Tok
from tweat.stream import MySQLInterface as sql
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sys, csv,numpy as np, random
db=sql('localhost','tweat','tweat','gun_sample')
T=Tok(preserve_case=False, strip_handles=False)
words_the={}
words_gun={}
it=0
us_en=open('words_alpha.txt').read().split('\n')
_vectorizer = CountVectorizer(max_df=0.5, min_df=2.5e-5,
                                   stop_words='english',
                                   ngram_range=(2,2))
samples_all=[]
labels=[]
txt=db.query('SELECT TEXT FROM CONTROL_TWEETS ORDER BY RAND() LIMIT 100000')
for tw in txt:
        t=tw[0]
        if len(t)<10:
                continue
        try:
                samples_all.append(t)
                labels.append(0)
        except:
                continue
with open(sys.argv[1], newline='') as txt2:
      for t in txt2:
        if len(t)<10:
                continue
        try:
                samples_all.append(t)
                labels.append(1)
        except:
                continue
with open(sys.argv[2], newline='') as txt2:
      for t in txt2:
        if len(t)<10:
                continue
        try:
                samples_all.append(t)
                labels.append(2)
        except:
                continue
tf = _vectorizer.fit_transform(samples_all)
kwds=_vectorizer.get_feature_names()
X=ComplementNB()
X.fit(tf, labels)
def all(L, l):
        for x in L:
                if not l(x):
                        return False
        return True
sorted_ind=np.argsort(X.feature_log_prob_[1])
for i in sorted_ind:
       label=X.predict(_vectorizer.transform([kwds[i]]))[0]
       if label!=0 and all(kwds[i].split(), lambda x: x in us_en):
              print(kwds[i], label)
       sys.stdout.flush()
