import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from tweat.stream import MySQLInterface
import json
output_file('vis_embed.html')

db=MySQLInterface('127.0.0.1','tweat','gg','tweat')
try:
    e=np.loadtxt('embed.np')
except OSError:
    weights_serialized=db.query("SELECT EMBEDDINGS FROM WORD2VEC ORDER BY ID DESC LIMIT 1")
    s_embeddings=weights_serialized[0][0]
    e=np.array(json.loads(s_embeddings))

vocab=[""]*e.shape[0]
indices=[]
vocab_list=(db.query("SELECT DISTINCT ID, WORD FROM WORDOCCURRENCES"))
for word in vocab_list:
    if not word[1].isalnum():
        continue
    if word[0]>=len(vocab):
        continue
    vocab[word[0]]=word[1]
    print(word[1])
def test_word_pair(x,y):
    i=vocab.index(x)
    j=vocab.index(y)
    return [np.dot(e[i], e[j])/(np.linalg.norm(e[i])*np.linalg.norm(e[j])), np.dot(e[i],e[j])]
#positive examples
print(test_word_pair("downturn", "recession"))
print(test_word_pair("leave", "betrayal"))
print(test_word_pair("europe", "eu"))
print(test_word_pair("merkel", "germany"))
print(test_word_pair("twitter", "facebook"))
print(test_word_pair("conscience", "right"))
print(test_word_pair("royal", "queen"))
print(test_word_pair("may", "minister"))
print(test_word_pair("brexit", "pound"))
print(test_word_pair("sterling", "pound"))
print(test_word_pair("squat", "pound"))
#negative examples
print(test_word_pair("lagu", "wanker"))
print(test_word_pair("brexit", "pics"))
print(test_word_pair("conscience", "rehash"))
print(test_word_pair("feckless", "cancer"))
#false positives (due to the sense ambiguity of "pound" and similarity to "stone" in british english)
#these also cause false negatives in the first set
#this sort of issue is unfortunately inevitable in such a metric embedding
print(test_word_pair("brexit", "stone"))
print(test_word_pair("sterling", "stone"))
print(test_word_pair("sterling", "squat"))

