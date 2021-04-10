import numpy as np
import tensorflow as tf
import nltk, json
from _mysql_exceptions import ProgrammingError as MySQLError, OperationalError as MySQLOpError
from ..stream import MySQLInterface
def cross_entropy(x,y):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(x), logits=y)
class MetricSpace:
    def __init__(self, db, vocab_size, sess):
        self.input=tf.placeholder(tf.float32, [None, vocab_size])
        self.label=tf.placeholder(tf.float32, [None, vocab_size])
        self.true_distance=tf.placeholder(tf.float32, [None, 1])
        self.weights=tf.Variable(tf.random_uniform([vocab_size, 50], -1.0,1.0))
        self.biases=tf.Variable(tf.zeros([50]))
        self.output=(tf.add(tf.matmul(self.input, self.weights),self.biases))
        self.expected=(tf.add(tf.matmul(self.label, self.weights), self.biases))
        self.h1=0.5*cross_entropy(self.expected,self.output)
        self.h2=0.5*cross_entropy(self.output,self.expected)
        self.h=tf.reduce_mean(tf.abs(tf.subtract(self.h1,self.h2)))
        self.metric_error=tf.abs(tf.subtract(self.h,self.true_distance))
        thresh=tf.constant(0.5)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.greater(self.h,thresh),tf.greater(self.true_distance,thresh)),tf.float32))
        self.optimizer=tf.train.GradientDescentOptimizer(1).minimize(self.metric_error)
        self.vocab_size=vocab_size
        self.db=db
        sess.run(tf.global_variables_initializer())
        try:
            try:
                ret=self.load(sess)
                if ret==0:
                    return
            except MySQLError:
                pass
        except MySQLOpError:
            pass
        db.execute("CREATE TABLE ica (id INT, weights LONGTEXT, biases LONGTEXT)")
        self.save(sess)
    def save(self, sess):
        self.w=sess.run(self.weights)
        self.b=sess.run(self.biases)
        idents=self.db.query("SELECT id FROM ica")
        if len(idents)>0:
            ident=int(idents[0][0])
        else:
            ident=0
        self.db.execute("INSERT INTO ica VALUES (%i, %%s, %%s)"%(ident+1), json.dumps(self.w.tolist()), json.dumps(self.b.tolist()))
        self.db.execute("DELETE  FROM ica WHERE id < %i"%(ident+1))
    def load(self,sess):
        W=self.db.query("SELECT weights, biases FROM ica")
        if len(W)<=0:
            return -1
        self.w=np.array(json.loads(W[0][0]))
        self.b=np.array(json.loads(W[0][1]))
        sess.run(tf.assign(self.weights,self.w))
        sess.run(tf.assign(self.biases,self.b))
        return 0
    def train(self, sess, tweet_pairs, distance=None):
        tokenizer=nltk.tokenize.TweetTokenizer()
        one_hot_input=np.zeros((len(tweet_pairs),self.vocab_size))
        one_hot_label=np.zeros((len(tweet_pairs),self.vocab_size))
        for i,tweet_pair in enumerate(tweet_pairs):
            in_put=tokenizer.tokenize(tweet_pair[0])
            label=tokenizer.tokenize(tweet_pair[1])
            for w in in_put:
                one_hot_input[i,hash(w)%self.vocab_size]=1
            for w in label:
                one_hot_label[i,hash(w)%self.vocab_size]=1
        if distance is not None:
            dist_np=np.array(distance).reshape((len(distance),1))
            feed={self.true_distance: dist_np, self.input: one_hot_input, self.label: one_hot_label}
            _,loss,h,acc=sess.run([self.optimizer, self.metric_error,self.h,self.accuracy], feed_dict=feed)
            return loss,h,acc
        else:
            feed={self.input: one_hot_input, self.label: one_hot_label}
            X,acc = sess.run([self.h,self.accuracy], feed_dict=feed)
            return X,acc



import sys
def train_metric(*args, train=True):
    db=MySQLInterface(*args)
    session=tf.Session()
    metric=MetricSpace(db,14000,session)
    distance=[]
    tweet_pairs=[]
    t=1
    for line in sys.stdin:
        if t%100==0:
            distance=[]
            tweet_pairs=[]
        if line.strip()=='':
            continue
        tweet_pairs.append(line.split("\t|\t")[:2])
        distance.append(float(line.split()[-1]))
        t+=1
        if t%100==0:
            if train:
                loss,h,acc=metric.train(session, tweet_pairs, distance)
#                print(h,distance)
                print(acc)
            else:
                d,acc=metric.train(session, tweet_pairs, None)
                print(ac)
    metric.save(session)

