import tensorflow.compat.v1 as tf
from sklearn.base import BaseEstimator, TransformerMixin
tf.disable_eager_execution()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import random, io, sys, math, json
from ..stream import MySQLInterface
from ..tokenize import Tokenizer
from ..parse import parse
from MySQLdb._exceptions import ProgrammingError as MySQLError, OperationalError as MySQLOperationalError

skip=set(stopwords.words('english'))

class Word2VecModel(BaseEstimator, TransformerMixin): 

    def __init__(self, embedding_size=50, num_sampled=40000, batch_size=16, learn_rate=0.25, is_remote = False, vocabulary_size=1024*1024, e=None, fname=None, corpus=None, keep_training=True):
        self.keep_training=True
#        self.cluster = tf.train.ClusterSpec({"worker": ["50.180.207.63:2222"]})
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.num_sampled=num_sampled
        self.batch_size=batch_size
        self.embeddings = tf.Variable(tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(
                  tf.random.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.input = tf.placeholder(tf.int32, shape=[None,15])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size,1 ])
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        self.embed2 = tf.nn.embedding_lookup(self.embeddings, self.input, validate_indices=False)
        self.loss=loss = tf.reduce_mean(
                                        tf.nn.sampled_softmax_loss(
                                            weights=self.nce_weights,
                                            biases=self.nce_biases,
                                            labels=self.train_labels,
                                            inputs=self.embed,
                                            num_sampled=self.num_sampled,
                                            num_classes=self.vocabulary_size,
                                            partition_strategy="div")
                                                                              )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(self.loss)
        self.W=None
        self.b=None
        self.e=None
        if fname is not None:
            js=json.loads(open(fname).read())
            self.W=np.array(js[0])
            self.b=np.array(js[1])
            self.e=np.array(js[2])
            self.fname=fname 
        else:
            self.e=e
            self.fname='w2v_out.js'
        init_op=tf.global_variables_initializer()
        config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12} , log_device_placement=False)
        sess = tf.Session(config=config)
        sess.run(init_op)
        self.session=sess
        self.load(self.session)
        if corpus!=None:
                self.id_all_words(corpus) 
    def load(self, session):
        if self.W is not None:#if we have load previous embeddings, weights, biases from the database...
            session.run(tf.assign(self.nce_weights, self.W))
            session.run(tf.assign(self.nce_biases, self.b))
            session.run(tf.assign(self.embeddings, self.e))
            self.glovew=open('glove_words_only.txt','r').split()
            self.words=open('%s_words.txt'%self.fname,'r').split('\n')
        elif self.embedding_size in [50,100,200]:
            self.e=np.zeros((self.vocabulary_size,self.embedding_size))
            glove=open('glove.twitter.27B.%id.txt'%self.embedding_size)
            self.glovew=[]
            ps=PorterStemmer()
            for i,line in enumerate(glove):
                if i>20000:
                        break
                if line.strip()!='':
                        fields=line.split()
                        vec=np.array(list(map(float,fields[-self.embedding_size:])))
                        self.e[i,:]=vec
                        self.glovew.append(ps.stem(fields[0].strip().lower()))
            open('glove_words_only.txt','w+').write('\n'.join(self.glovew))
            self.words=[w for w in self.glovew]
            session.run(tf.assign(self.embeddings, self.e))
    def fit(self,X, y=None):
        skip_window=5
        if self.keep_training==False:
                return self
        session=self.session
        batch_labels=[]
        batch_inputs=[]
        T=Tokenizer()
        for epoch in range(100):
                #keep looking through tweets until we have a whole training batch
                while len(batch_inputs)<self.batch_size:
                            tokens=list(map(self.gettokenid,random.choice(X)))
                            if len(tokens)<skip_window+1:
                                continue
                            for i,tok0 in enumerate(tokens[:-skip_window-1]):
                                        tok1=(tokens[i+skip_window])
                                        if max(tok0,tok1)>=self.vocabulary_size or min(tok1,tok0)<0:
                                                continue
                                        batch_labels.append(tok0)
                                        batch_inputs.append(tok1)
            
                            if len(batch_inputs)>=self.batch_size:
                                break
                avg_loss=0

                output=[self.optimizer,self.loss,self.nce_weights, self.nce_biases, self.embeddings]

                feed={self.train_inputs: np.array(batch_inputs[:self.batch_size]), self.train_labels: np.array(batch_labels[:self.batch_size]).reshape((self.batch_size,1))}
                batch_inputs=batch_inputs[self.batch_size:]
                batch_labels=batch_labels[self.batch_size:]
                
                output=[self.optimizer,self.loss,self.nce_weights, self.nce_biases, self.embeddings]
                _,cur_loss,self.W,self.b,self.e=session.run(output,feed_dict=feed)
                sys.stderr.write('%f\n'%cur_loss)
        sys.stderr.flush()
        self.save(self.fname)
        return self
    def save(self, fname):
        s_weights=io.BytesIO()
        open('fname', 'w+').write(json.dumps([self.W.tolist(),self.b.tolist(),json.dumps(self.e.tolist())]))
    def transform(self,items):
               X=np.array([[self.gettokenid(tok) if self.gettokenid(tok)>=0 else 0 for tok in toks.split()] for toks in items])
               embed= self.session.run([self.embed2], feed_dict={self.input:X})[0]
               return embed.reshape([-1,X.shape[-1]*self.embedding_size])
    def gettokenid(self,w):
        if w in self.words:
                return self.words.index(w)
        return -1 
    def id_all_words(self,corpus):
        for T in corpus:
                for w in T.split():
                        if len(self.words)>=self.vocabulary_size:
                                return
                        if w not in self.words:
                                self.words.append(w)
                                
                                
class Word2Vec:
    def __init__(self, db, sess=None):
        self.embeddings=tf.Variable(np.array(json.loads(serialized[0][0])))
        self.input=tf.placeholder(tf.int32, shape=[None,1])
        self.N=self.embeddings.get_shape().as_list()[0]
        self.embed=tf.nn.embedding_lookup(self.embeddings, self.input, validate_indices=False)
        self.session=sess
        if sess is None:
            self.session=tf.Session()
        self.session.run(tf.global_variables_initializer())
    def __call__(self, words):
         session=self.session
         wordids=[]
         for w in words:
             if w is not str:
                wordids.append(w)
             else:
                 wid=self.get_token_id(w)
                 if wid>=0:
                    wordids.append(wid)
         index_label=np.array(wordids).astype(np.int32)
         index_label[index_label>=self.N]=0
         index_label=index_label.reshape((index_label.size,1))
         embed,= self.session.run([self.embed], feed_dict={self.input:index_label})
         return embed       



def word2vec_all(N=1000, bsize=1000, RunRemote=False):
#    with tf.device('/device:GPU:0'):
        pargs = parse(topic=False)
        db=MySQLInterface(pargs['host'],pargs['user'],pargs['password'],pargs['db'])
        w2vec=Word2VecModel(is_remote = RunRemote, batch_size=bsize)
        init_op=tf.global_variables_initializer()
        config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 12} , log_device_placement=False)
        sess = tf.Session(config=config)
        sess.run(init_op)
        w2vec.load(sess)
        for t in range(N):
            print(w2vec.train(sess))
            sys.stdout.flush()
        w2vec.save()
