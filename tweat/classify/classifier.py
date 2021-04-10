import numpy as np
from ..stream import MySQLInterface
from sklearn.ensemble import RandomForestClassifier as Regression
from ..tokenize import tokenize_tweet
import sys
from bisect import bisect_right, bisect_left
from ..tokenize import tokenize_tweet, Tokenizer
w2v=None
wids=[]
Tok=Tokenizer()
def freq(x,wids):
    return bisect_right(wids,x)-bisect_left(wids,x)
def encode(w,wids):
    return freq(w,wids)
def vector_encode(w,w2v):
    W=w2v(w)
    return W/np.linalg.norm(W)
def stringify(words,db):
    string=sorted(list(map(encode,words)))[:10]
    return string
def vectorize(words,db,w2v,wids):
    freq=np.array([encode(w,wids) for w in words])
    string=np.array(words)[np.argsort(freq)]
    S=vector_encode(string[:8],w2v).flatten().tolist()
    return S
def train_model(db, produce_test_data,w2v,wids):
    anti_list=db.query('SELECT TOK,LABEL FROM TWEETSAUTOLABELED WHERE LABEL=-1 ORDER BY RAND()')
    pro_list=db.query('SELECT TOK,LABEL FROM TWEETSAUTOLABELED WHERE LABEL=1 ORDER BY RAND()')
    test_data=[]
    test_labels=[]
    test_tweets=db.query("SELECT T.TEXT,L.LABEL FROM TWEETSLABELED L INNER JOIN TWEETSTOKENIZED T ON T.ID=L.ID order by RAND();")
    for tweet in test_tweets:
        test_data.append(tweet)
        test_labels.append(int(tweet[1])>2)
    encoded=[]
    tweetlabels=[]
    for i,tweet in enumerate(list(anti_list)+list(pro_list)+test_data[:-100]):
        if len(tweet[0].split())>=8:
            words=list(map(int,tweet[0].split()))
            st=vectorize(words,db,w2v,wids)
            encoded.append(st)
            tweetlabels.append(int(tweet[1])>2 if i>=len(anti_list+pro_list) else int(tweet[1])>0)
    Classifier=Regression()
    Classifier.fit(np.array(encoded),tweetlabels)
    print(len(encoded))
    if produce_test_data:
        return Classifier, test_data[-100:], test_labels[-100:]
    else:
        return Classifier
def get_words(*args):
    global wids;
    db=MySQLInterface(*args[0:4])
    F=open('words.txt')
    english_words=[x.strip().lower() for x in F]
    words=sorted(db.query("SELECT ID,WORD FROM WORDOCCURRENCES"), key=lambda x: int(x[0]))
    wids=[int(x[0]) for x in words]
    words=set(words)
    for w in words:
        if w[1].strip().lower() not in english_words:
            continue
        print(w[1], freq(int(w[0])))
def gen_manual_test_data():
    test_tweets=db.query("SELECT T.TEXT,L.TEXT FROM TWEETS L INNER JOIN TWEETSTOKENIZED T ON T.ID=L.ID order by RAND() limit 150;")
    texts=[]
    labels=[]
    for tweet in test_tweets:
        print(tweet[0][0])
        texts.append(tweet[0])
        if int(input())==1:
            labels.append(0)
        else:
            labels.append(1)
    return texts, labels

def test_classify(*args):
    from ..word2vec import Word2Vec
    global wids;
    #global wordtxts;
    global w2v;
    db=MySQLInterface(*args[0:4])
    words=db.query("SELECT TEXT FROM TWEETSLABELED")+db.query("SELECT TEXT FROM TWEETSAUTOLABELED")
    wids+=sorted([int(x[0]) for x in words])
    w2v=Word2Vec(db)
    labels=[]
    err=0
    N=0
    err_minus=0
    acc_plus=0
    acc_minus=0
    err_plus=0
    C, _,_=train_model(db,True,w2v,wids)
    tweets,labels=gen_manual_test_data()
    encoded=[]
    Ntrials=len(tweets)
    for i in range(Ntrials):
        tweet=tweets[i][0].split()
        if len(tweet)<8:
            continue
        tokens = list(map(int,tweet))
        label=labels[i]
        tweet_enc=vectorize(tokens,db,w2v,wids)
        fake_label=C.predict(np.array(tweet_enc).reshape([1,-1]))[0]
        N+=1
        if int(label)!=int(fake_label):
            err+=1
            if label:
                err_minus+=1
            else:
                err_plus+=1
            print('error on tweet labeled', label, "; classifier predicted ",fake_label)
        elif label:
            acc_plus+=1
            print('success on tweet labeled', label,  "; classifier predicted ",fake_label)
        else:
            acc_minus+=1
            print('success on tweet labeled', label, "; classifier predicted ",fake_label)
        print('running error tally: %f%%'%(err/N*100))
    print(' total error rate:',err/N*100, '%%')
    tpr=acc_plus/(acc_plus+err_minus)*100
    fnr=err_minus/(acc_plus+err_minus)*100
    fpr=err_plus/(acc_minus+err_plus)*100
    print(' true positive rate:',tpr,'%%')
    print(' false negative rate:',fnr,
            '%%')
    try:
        precision=tpr/(tpr+fpr)
        recall=tpr/(tpr+fnr)
        f1=2*(precision*recall)/(precision+recall)
        print(' precision:',precision)
        print(' recall:',recall)
        print('f1-score:', f1)
    except ZeroDivisionError:
        pass
    print('percent accuracy:',100-err*100/N, '%%')
