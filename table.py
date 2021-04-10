from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
#from gensim.sklearn_api.tfidf import TfIdfTransformer as TfidfTransformer
import gensim, nltk
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tweat.word2vec import Word2VecModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
import pyhash
import re
URL_REGEX = re.compile(
    u"^"
    # protocol identifier
    u"(?:(?:https?|ftp)://)"
    # user:pass authentication
    u"(?:\S+(?::\S*)?@)?"
    u"(?:"
    # IP address exclusion
    # private & local networks
    u"(?!(?:10|127)(?:\.\d{1,3}){3})"
    u"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    u"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    u"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    u"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    u"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    u"|"
    # host name
    u"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    u"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    u"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
    u")"
    # port number
    u"(?::\d{2,5})?"
    # resource path
    u"(?:/\S*)?"
    u"$"
    , re.UNICODE)
class Hasher(BaseEstimator, TransformerMixin):
        def __init__(self,Nwords=100000):
                self.hasher= pyhash.fnv1_32()
                self.Nwords=Nwords
        def fit(self,X, y=None):
                return self
        def transform(self,X):
                return [[(self.hasher(y)%self.Nwords,1) for y in x] for x in X]

class Reshaper(BaseEstimator, TransformerMixin):        
        def __init__(self, Nwords=100000):
                self.Nwords=Nwords
        def fit(self,X, y=None):
                return self
        def transform(self,X):
                return gensim.matutils.corpus2csc(X, num_terms=self.Nwords).T
import numpy as np
from nltk.stem import PorterStemmer
import matplotlib.path as mpltPath
import sys, random, re
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
ps=PorterStemmer()
def gen_state_polys():
        tree = ET.parse('states.xml')
        root = tree.getroot()
        polys={}
        for child in root:
                if child.tag!='state':
                        continue
                plist=[]
                for p in child:
                        if p.tag!='point':
                                continue
                        plist.append([float(p.attrib['lng']), float(p.attrib['lat'])])
                polys[child.attrib['name'].lower().strip()]=np.array(plist)
        return polys
def state_has_p(state_bounds, lnglat):
        path = mpltPath.Path(state_bounds)
        return path.contains_points(lnglat)
def longlat2state(all_state_bounds,lnglat):
        for s in all_state_bounds:
                if state_has_p(all_state_bounds[s],np.array([lnglat])):
                        return s
        return None
#for j in range(12):
places=open('us-cities.txt').read().split('\n')
def not_a_place(x):
        for p in places:
                if p.strip().lower() in x.lower():
                        return False
        return True
from sklearn.naive_bayes import MultinomialNB
if True:
#        U=open('libs3.txt')
#        libs2=[x.strip() for x in U]
#        U2=open('cons3.txt')
#        cons2=[x.strip() for x in U2]
        bad_words=open('hashtags_ag.txt').read().split()
        good_words=open('hashtags_pg.txt').read().split()
        from matplotlib import pyplot as plt
        #from tweat.stream import MySQLInterface as sql
        #        db=sql(*sys.argv[2:])
#        tweets_from_db=db.query('SELECT ID,TEXT,USER FROM TWEETS') 
#        T=sorted(tweets_from_db,key=lambda x: int(x[0]))
#        users=db.query('SELECT USER,LOCATION FROM USERS')
#        users_loc={u:v for u,v in users}
#        ids=[int(x[0]) for x in T]
        sys.stderr.write('done sorting\n')
        sys.stderr.flush()
        alls=gen_state_polys()
        kwreg={s:0 for s in alls}
        kwfear={s:0 for s in alls}
        #gfilters=["gun violence", "gun rights", "second amendment", "assault weapons", " #2A ", "2nd amend", "guncontrol", "gun control", "mass shooting", "gun owner"]
        #labels=[2,1,1,0,1,1,1,2,2,0,0]
#        w1=[gfilters[w] for w in range(len(gfilters)) if labels[w]==1]
#        w2=[gfilters[w] for w in range(len(gfilters)) if labels[w]==2]
#        filters=open('twitter_dict.txt','r')
#        w1=filters.readline().split(',')
#        w2=filters.readline().split(',')
#        dict_filters1=re.split(',|&',open('proquest_labels_survey_1.csv','r').read())
#        dict_filters2=re.split(',|&', open('proquest_labels_survey_2.csv','r').read())

#        w1=["defense"]
#        w2=["rights"]
        w2=[('firearm','law'),('firearm','regulation'),('bills',), ('ban',), ('legislation',)]
        w1=[('burglar',),('gangs',)]
        sys.stderr.flush()
#        w1=["defense"]
#        w2=["rights"]
        sw={s:{i:0 for i in w1+w2} for s in alls}
        it=0
        total={s:0 for s in alls}
        #lines=open(sys.argv[1]).read().split('\n')
        from nltk.tokenize.casual import TweetTokenizer as tokenizer
        Tok=tokenizer()
        W=stopwords.words('english')
        days_in_month=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        j=7
        sent=[]
        corpus=[]
        corpus_labeled=[]
        labels=[]
        ix=0
        corpus=open('corpus.txt', 'r').read().split('\n')
        corpus_labeledf=open('labels4_uniq.txt', 'r')
        corpus_labeledf2=open('labels5.txt', 'r')
        sys.stderr.write('done tokenizing...')
        sys.stderr.flush()
        corpus_train=[]
        corpus_test=[]
        labels_train=[]
        labels_test=[]
        for j,f in enumerate(corpus_labeledf):
                fields=f.lower().replace(';',' ').split()
                if len(fields)==0:
                        continue
                if fields[-1]=='??' or fields[-1]=='na':
                        fields[-1]='na'
                        continue
                toks=nltk.tokenize.TweetTokenizer().tokenize(' '.join(fields[:-1]))
                if len(toks)<8:
                        continue
                corpus_labeled.append(URL_REGEX.sub('',' '.join(toks)))
                labels.append(fields[-1])

        sys.stderr.write("labeled"+ " "+str(len(labels))+'\n')
        sys.stderr.flush()
        corpus_train,corpus_test,labels_train,labels_test=train_test_split(corpus_labeled,labels,test_size=0.1, random_state=52)


        for j,f in enumerate(corpus_labeledf2):
                fields=f.lower().replace(';',' ').split()
                if len(fields)==0:
                        continue
                if fields[-1]=='??' or fields[-1]=='na':
                        fields[-1]='na'
                        continue
                toks=nltk.tokenize.TweetTokenizer().tokenize(' '.join(fields[:-1]))
                if len(toks)<8:
                        continue
                corpus_train.append(URL_REGEX.sub('',' '.join(toks)))
                labels_train.append(fields[-1])
     
        clf=Pipeline([('h',CountVectorizer(ngram_range=(2,2), tokenizer=lambda x: x.split())), 
                        ('t',TfidfTransformer()),
#                         ('c', LogisticRegression(multi_class='ovr', solver='sag'))])
                         ('c', RandomForestClassifier(8*4))])

        states=sorted([s for s in alls])


        clf.fit(corpus_train, labels_train)

        classify = lambda x: clf.predict(x)

        pred=classify(corpus_test)
        sys.stderr.write('%i'%np.sum(np.array(labels_test)=='pg')+'\n')
        sys.stderr.write('%i'%np.sum(np.array(labels_test)=='ag')+'\n')
        sys.stderr.write('%i'%(np.sum(np.array(labels_test)=='na'))+'\n')
        sys.stderr.write(str(accuracy_score(np.array(labels_test), pred))+'\n')
        sys.stderr.write(str(precision_recall_fscore_support(np.array(labels_test), pred ))+'\n')
        sys.stderr.flush()
        sys.stderr.write('done training\n')
        sys.stderr.flush()
#        sys.exit(-1)
        flines=open('all_tweets_seq_sorted3.csv', 'r')
        last_month=5
        month=1
        kwfear_m={s:0 for s in states}
        kwreg_m={s:0 for s in states}
        total_m={s:0 for s in states}
        kwfear_d=[0]*365
        kwreg_d=[0]*365
        total_d=[0]*365
        clff=open('corpus_plus.txt', 'w+')
        for linei in flines:
                it+=1
                #if sys.argv[-1] in linei:
                #        continue
                if linei.strip()=='':
                        continue
                a=linei.split()
                try:
                        time=float(a[0])
                except ValueError:
                        continue
                month=np.where(time<np.cumsum(24*60*60*days_in_month))[0][0]
                day=int(time/(24*60*60))        
                if month!=last_month:
                        sys.stderr.write('%i\n'%month)
                        sys.stderr.flush()
                        f2=open('table_m%i.csv'%(month), 'w+')
                        for s in states:
                                if total_m[s]==0:
                                
                                        f2.write(s+ " %f %f\n"%(0, 0))
                                else:
                                        f2.write(s+ " %f %f\n"%( kwfear_m[s]/total_m[s], kwreg_m[s]/total_m[s]))
                        total_m[s]=0
                        kwfear_m[s]=0
                        kwreg_m[s]=0
                        total_m[s]=0
                        last_month=month
                if len(a)<3:
                        continue
                s=a[1].replace('_',' ')
                toks=nltk.tokenize.TweetTokenizer().tokenize(' '.join(a[6:]).lower())
#                if classify([' '.join(toks)])[0]!='pg':
#                      sys.stderr.write(str(classify([' '.join(toks)])[0])+'\n')
#                       continue
#                else:
#                        clff.write(('%s %i %s\n'%(s, int(time), ' '.join(toks))))
                if day>181 and day<=305:
                        total[s]+=len(toks)
                total_m[s]+=len(toks)
                total_d[day]+=len(toks)
                S=' '.join(toks)
                found_bad=False
                found_good=False
                for w in bad_words:
                        if '#'+w.lower().strip() in S:
                                found_bad=True
                                break
                for w in good_words:
                        if '#'+w.lower().strip() in S:
                                found_good=True
                                break
                if not found_good or found_bad:
                        continue
                def all_in(word_tuple,string):
                        for w in word_tuple:
                                if w not in string:
                                        return False
                        return True
                for w in w1+w2:
                        if all_in(w,S):
                                if day>181 and day<=305:
                                               sw[s][w]+=1
                for w in w1:
                        if all_in(w,S):
                                kwfear_m[s]+=1
                                kwfear_d[day]+=1
                                if day>181 and day<=305:
                                        kwfear[s]+=1
                for w in w2:
                        if all_in(w,S):
                                kwreg_m[s]+=1
                                if day>181 and day<=273:
                                        kwreg[s]+=1
                                kwreg_d[day]+=1
                sys.stderr.write(str(it)+' '+str(sum(kwfear.values()))+' '+str(sum(kwreg.values()))+'\n')
                sys.stderr.flush()
         
        states=sorted([s for s in alls])
        wc=np.zeros((len(w1+w2),len(states)))
        c1=np.array([float(kwfear[s]/total[s]) if total[s]>0 else 0 for s in states])
        c2=np.array([float(kwreg[s]/total[s]) if total[s]>0 else 0 for s in states])
        #K=c1+c2
        #c1/=K
        #c2/=K
        for i,s in enumerate(states):        
                for j2,w in enumerate(w1+w2):
                        wc[j2,i]=sw[s][w]/total[s]
        for i,a in enumerate(wc):
                print((w1+w2)[i],np.sum(a),np.corrcoef(a,c1)[0,1], np.corrcoef(a,c2)[0,1], 1 if i<len(w1) else 2)
        #        plt.figure()
        #        plt.title((w1+w2)[i])
        #        plt.scatter(a,c1, label='w/ bucket 1')
        #        plt.scatter(a,c1, label='w/ bucket 2')
        #        plt.legend()
        #        plt.savefig((w1+w2)[i]+'.png')
        for i,s in enumerate(states):
                print(s,c1[i],c2[i])
                sys.stdout.flush()
        for i in range(365):
                print(i, kwreg_d[i], total_d[i])
