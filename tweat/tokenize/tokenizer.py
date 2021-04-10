from ..stream import MySQLInterface
from MySQLdb._exceptions import ProgrammingError as MySQLError, OperationalError as MySQLOperationalError
from nltk import TweetTokenizer as Tokenizer
def tokenize_all(parsed, debug_print=False, location_only=False):
    host=parsed['host']
    username=parsed['user']
    pwd=parsed['password']
    dbname=parsed['db']
    #fixme: make this pausable/resumable...
    tokenizer=Tokenizer()
    db=MySQLInterface(host, username, pwd, dbname)
    try:
        db.query("SELECT * FROM TWEETSTOKENIZED LIMIT 1")
        db.query("SELECT * FROM WORDOCCURRENCES LIMIT 1")
    except MySQLError:
        try:
            db.execute("DROP TABLE TWEETSTOKENIZED;")
        except:
            pass
        try:
            db.execute("DROP TABLE WORDOCCURRENCES;")
        except:
            pass
        db.execute('CREATE TABLE TWEETSTOKENIZED (ID INT, USER VARCHAR(16), TOPIC VARCHAR(512), ISODATE VARCHAR(512), PLACE VARCHAR(4096), TEXT LONGTEXT)')
        db.execute('CREATE TABLE WORDOCCURRENCES (ID BIGINT, WORD LONGTEXT) CHARACTER SET UTF8 COLLATE utf8_unicode_ci')
    tweets=db.query('SELECT * FROM TWEETS WHERE ID NOT IN (SELECT ID FROM TWEETSTOKENIZED)'+(' AND PLACE!="1"' if location_only else '')+"ORDER BY RAND()")
    for tweet in tweets:
        txt=tweet[5]
        ID=tweet[0]
        tweet=[ID,tweet[1],tweet[2],tweet[3],tweet[4],'\n'.join(tokenizer.tokenize(txt.replace('\n',' ').replace('\\n',' ').replace('#', '# ').lower()))]
        wordids=[]
        for token in tweet[5].split('\n'):
            #add the word to the table of integerized words if it doesn't already exist there
            index=db.query('SELECT ID FROM WORDOCCURRENCES WHERE WORD=%s LIMIT 1',token)
            word_id=None
            if index is not None:
                if len(index)>=1:
                    word_id=int(index[0][0])
            if word_id is None:
                word_id=int(db.query('SELECT COUNT(DISTINCT WORD) FROM WORDOCCURRENCES')[0][0])

            db.execute('INSERT INTO WORDOCCURRENCES VALUES (%i,%%s)'%(word_id), token) 
            wordids.append(str(word_id))       
        tweet[5]='\n'.join(wordids)
        db.execute('INSERT INTO TWEETSTOKENIZED VALUES (%i,%%s,%%s,%%s,%%s,%%s)'%int(ID),*tweet[1:])        


def tokenize_tweet(txt,*args, **kwargs):
    tokenizer=Tokenizer()
    if len(args)>1:
        db=MySQLInterface(*args)
    else:
        db=args[0]
        
    txt_res='\n'.join(tokenizer.tokenize(txt.replace('\n',' ').replace('\\n',' ').lower()))
    wordids=[]
    for token in txt_res.split('\n'):
            #add the word to the table of integerized words if it doesn't already exist there
            index=db.query('SELECT ID FROM WORDOCCURRENCES WHERE WORD=%s LIMIT 1',token)
            word_id=None
            if index is not None:
                if len(index)>=1:
                    word_id=int(index[0][0])
            if word_id is None:
                word_id=int(db.query('SELECT COUNT(DISTINCT WORD) FROM WORDOCCURRENCES')[0][0])

            db.execute('INSERT INTO WORDOCCURRENCES VALUES (%i,%%s)'%(word_id), token) 
            wordids.append((word_id))
    return wordids

def tokenize_user(user, db):
      all_txt=db.query('SELECT * FROM TWEETS WHERE USER=\'%s\''%user)
      wordlist=[0]*(int(db.query('SELECT ID FROM WORDOCCURRENCES ORDER BY ID DESC LIMIT 1;')[0][0])+1)
      all_words=db.query('SELECT ID,WORD FROM WORDOCCURRENCES');
      tokenizer=Tokenizer()
      for w in all_words:
          wordlist[int(w[0])]=w[1]
      print("word list loaded")
      for twt in all_txt:
            tweet=list(twt)
            txt=tweet[5]
            txt_res='\n'.join(tokenizer.tokenize(txt.replace('\n',' ').replace('\\n',' ').lower()))
            wordids=[wordlist.index(x) for x in txt_res if x in wordlist]
            tweet[5]='\n'.join([str(x) for x in wordids])
            db.execute('INSERT INTO TWEETSTOKENIZED VALUES (%i,%%s,%%s,%%s,%%s,%%s)'%int(tweet[0]),*tweet[1:])        
