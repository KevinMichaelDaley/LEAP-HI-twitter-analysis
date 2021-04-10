import MySQLdb as sql
import sys, os
from string import punctuation
db=sql.connect(*sys.argv[1:])
N=5000 #2000000 characters / mo. free from bing
c=db.cursor()
from mstranslator import Translator
char_count=0
from nltk.tokenize import TweetTokenizer as Tokenizer
def transform_to_words(s):
    t=Tokenizer()
    ss=t.tokenize(s)
    ws=[w for w in ss if (w.isalnum() or w in punctuation)] #this will exclude urls, which are a mixture.
    return ' '.join(ws)
        
with open('training_examples.txt', 'a+', encoding='utf-8') as f:
    T=Translator(os.getenv('translate_key'))
    while char_count<N:
       c.execute("SELECT TEXT FROM TWEETS WHERE TOPIC=':)' AND TEXT LIKE '% and %' ORDER BY RAND() LIMIT 1")
       tweet=c.fetchall()[0][0].replace('\n',' ')
       tweet=transform_to_words(tweet)
       c.execute("SELECT TEXT FROM TWEETS WHERE TOPIC=':(' AND TEXT LIKE '% and %' ORDER BY RAND() LIMIT 1")
       
       
       tweet2=c.fetchall()[0][0].replace('\n',' ') 
       tweet2=transform_to_words(tweet2)

       fw=T.translate(tweet,lang_to='zh')
       pos=T.translate(fw,lang_to='en', lang_from='zh')
       char_count+=len(fw)+len(tweet)
       print(pos+'\t|\t'+tweet+' 0.1'+'\n')
       print(tweet2+'\t|\t'+tweet+' 1.0'+'\n')
  
