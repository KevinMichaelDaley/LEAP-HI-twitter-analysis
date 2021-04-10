import sys
import datetime
import numpy as np
from tweat.stream import MySQLInterface
from tweat.classify import classifier
from tweat.tokenize import Tokenizer
from MySQLdb._exceptions import ProgrammingError as MySQLError
import time
def reconstruct_timeseries():
    db=MySQLInterface(*sys.argv[1:])
    tokenizer=Tokenizer()
    fusers=open("users.txt","r")
    def datekey(x):
        return datetime.datetime(*time.strptime(x, "%Y-%m-%dT%H:%M:%S")[:6])
    dates=[]
    def datetime_to_day(d):
        epoch = datetime.datetime(2019, 1, 1)
        if epoch>d:
            return -1
        day =  (d - epoch).total_seconds()/(604800/7)
        return int(day)
    i=0
    fout1=open("timeseries_label.np","w+")
#    fout2=open("timeseries_entropy.np","w+")
    N=int(db.query("SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME=\"WORDOCCURRENCES\"")[0][0])
    for line1 in fusers:
        line=line1.replace("\n","")
        user_tweets=np.zeros(365)
        user_tweetsN=np.zeros(365)
        print(line)
        
        utweets=db.query("SELECT DISTINCT ISODATE, LABEL, ID from TWEETS_CLASSIFIED WHERE USER=\"%s\""%line)
#        tweets=db.query("SELECT DISTINCT ID FROM TWEETS WHERE USER = \"%s\""%line)
        if len(utweets)==0:
            fout1.write("nan "*365+"\n")
#            classifier.do_user(*sys.argv[1:], user=line)
#            utweets=db.query("SELECT DISTINCT ISODATE, LABEL, ID from TWEETS_CLASSIFIED WHERE USER=\"%s\""%line)
        print(utweets)
 #       entropy=np.zeros(365)
        for tweet in utweets:
            j=datetime_to_day(datekey(tweet[0]))
            print(j)
            if j<0:
                continue
 #           text=db.query("SELECT TEXT FROM TWEETS WHERE ID=%i"%int(tweet[2]))[0][0]
 #           for w in tokenizer.tokenize(text):
 #              try:
 #                   P=float(db.query("SELECT COUNT(WORD) FROM WORDOCCURRENCES WHERE WORD=\"%s\" GROUP BY WORD"%w)[0][0])/N
 #               except MySQLError:
 #                   P=0
 #               if P>0:
 #                   entropy[j]-=P*np.log(P)/np.log(2)

            user_tweetsN[j]+=1.0
            user_tweets[j]+=2*float(tweet[1])-1
        for j in range(365):
            if user_tweetsN[j]>0:
                user_tweets[j]/=user_tweetsN[j]
        fout1.write(" ".join([str(x) for x in user_tweets])+"\n")
#        fout2.write(" ".join([str(x) for x in entropy])+"\n")
        fout1.flush()
#        fout2.flush()
if __name__=="__main__":
    reconstruct_timeseries()
        
