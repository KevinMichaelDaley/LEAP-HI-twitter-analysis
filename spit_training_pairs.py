import MySQLdb as sql
import sys
db=sql.connect(*sys.argv[1:])
c=db.cursor()
from translate import Translator
fw= Translator(to_lang="de")
bw= Translator(from_lang="de",to_lang="en")
with open('training_examples.txt', 'a+', encoding='utf-8') as f:
    while True:
       c.execute("SELECT TEXT FROM TWEETS WHERE TOPIC=':)' ORDER BY RAND() LIMIT 1")
       tweet=c.fetchall()[0][0]

       c.execute("SELECT TEXT FROM TWEETS WHERE TOPIC=':(' ORDER BY RAND() LIMIT 1")

       
       tweet2=c.fetchall()[0][0]
       pos=bw.translate(fw.translate(tweet))
       
       f.write(pos+'\t|\t'+tweet+' 0.1'+'\n')
       f.write(tweet2+'\t|\t'+tweet+' 1.0'+'\n')
  
