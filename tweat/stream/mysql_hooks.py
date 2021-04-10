import MySQLdb as msql
import datetime
from MySQLdb._exceptions import ProgrammingError as MySQLError
class MySQLInterface:
    def __init__(self, hostname, username, pwd, dbname):
        '''connect to a remote mysql database'''
        self.db_connection=msql.connect(host=hostname, user=username, passwd=pwd, db=dbname,  use_unicode=True, charset='utf8', init_command='SET NAMES UTF8')
        self.db_cursor=self.db_connection.cursor()

        self.execute("CREATE TABLE IF NOT EXISTS CONTROL_TWEETS (ID INT, USER LONGTEXT, TOPIC LONGTEXT, ISODATE LONGTEXT, PLACE VARCHAR(4096), TEXT LONGTEXT, RTTEXT LONGTEXT CHARACTER SET UTF8 COLLATE utf8_unicode_ci)")
        try:
            self.query("SELECT * FROM TWEETS LIMIT 1")
            self.query("SELECT * FROM TOPICS LIMIT 1")
            return
        except MySQLError:
            self.execute("CREATE TABLE TWEETS (ID INT, USER LONGTEXT, TOPIC LONGTEXT, ISODATE LONGTEXT, PLACE VARCHAR(4096), TEXT LONGTEXT, RTTEXT LONGTEXT CHARACTER SET UTF8 COLLATE utf8_unicode_ci)")
            self.execute("CREATE TABLE TOPICS (NAME LONGTEXT CHARACTER SET UTF8 COLLATE utf8_unicode_ci)")
    def query(self,query, *args):
        '''get stuff from the sql database'''
        self.db_cursor.execute(query,args)
        return self.db_cursor.fetchall()
    def iter_query(self,query, *args):
        '''get stuff from the sql database'''
        self.db_cursor.execute(query,args)
        while True:
            res_next=self.db_cursor.fetchone()
            if res_next is not None:
                yield res_next
            else:
                return
    def add_control_tweet(self,status, topic):
        place_id=status.place if status.place is not None else "b:"+(status.user.location if status.user.location is not None else "")
        ID=int(self.query('SELECT COUNT(ID) FROM CONTROL_TWEETS')[0][0])
        rt_text=""
        if hasattr(status, 'retweeted_status'):
            try:
               rt_text=status.retweeted_status.extended_tweet['full_text']
            except AttributeError:
               rt_text=status.retweeted_status.text
        try:
           full_text=status.extended_tweet['full_text']
        except AttributeError:
           full_text=status.text
        try:
            self.execute('INSERT INTO CONTROL_TWEETS VALUES(%i,%%s,%%s,%%s,%%s,%%s,%%s)'%(ID), status.user.screen_name, topic, status.created_at.isoformat(), str(place_id), full_text.encode('utf8'), rt_text.encode('utf8'))
        except:
            print("Error")
    def execute(self,command,*args):
        '''push commands to the sql database without getting stuff'''
        self.db_cursor.execute(command, args)
        self.db_connection.commit()
    def add_tweet(self,status, topic):
        place_id=status.place if status.place is not None else "b:"+(status.user.location if status.user.location is not None else "")
        ID=int(self.query('SELECT COUNT(ID) FROM TWEETS')[0][0])
        rt_text=""
        if hasattr(status, 'retweeted_status'):
            try:
               rt_text=status.retweeted_status.extended_tweet['full_text']
            except AttributeError:
               rt_text=status.retweeted_status.text
        try:
           full_text=status.extended_tweet['full_text']
        except AttributeError:
           full_text=status.text
        try:
            self.execute('INSERT INTO TWEETS VALUES(%i,%%s,%%s,%%s,%%s,%%s,%%s)'%(ID), status.user.screen_name, topic, status.created_at.isoformat(), str(place_id), full_text.encode('utf8'), rt_text.encode('utf8'))
        except:
            return
    def add_user_friends(self, user, friend):
                try:
                    try:
                        self.execute('INSERT INTO FRIENDS VALUES(\'%s\',\'%s\',\'%s\')'%(user, friend, datetime.datetime.now().isoformat()))
                    except UnicodeEncodeError:
                        pass
                except MySQLError:
                    self.execute("CREATE TABLE IF NOT EXISTS FRIENDS (FRIEND VARCHAR(16), FOLLOWER VARCHAR(16), ISODATE LONGTEXT CHARACTER SET UTF8 COLLATE utf8_unicode_ci)")
                    try:
                        self.execute('INSERT INTO FRIENDS VALUES(\'%s\',\'%s\',\'%s\')'%(user, friend, datetime.datetime.now().isoformat()))
                    except UnicodeEncodeError:
                        pass
    def query_tweets_by_place(self,place_id):
        '''does not return an iterator of statuses, rather an iterator of tuples in the db column format'''
        return self.query('SELECT * FROM TWEETS WHERE PLACE=?',place_id)
    def query_tweets_by_topic(self,topic):
        return self.query('SELECT * FROM TWEETS WHERE TOPIC=?',topic)
    def add_name(self, topic):
        self.execute('INSERT INTO TOPICS VALUES(%s)', topic)
    def query_names(self):
        return self.query('SELECT * FROM TOPICS GROUP BY NAME ORDER BY COUNT(*) DESC;')
