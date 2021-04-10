from tweat import stream
import sys
db=stream.MySQLInterface(*sys.argv[1:])
for tweet in db.iter_query('SELECT TOPIC, TEXT FROM TWEETS ORDER BY RAND() LIMIT 590'):
	if len(tweet[1])==0:
		continue
	if tweet[0]=='allergies':

		print(str(tweet[1].replace('\n',' ').replace('%','%%').encode('utf-8'))[2:].replace('\\','').replace('\'','').replace('[','').replace(']','').replace('[',''),0)
	elif tweet[0]=='rihanna':
		print(str(tweet[1].replace('\n',' ').replace('%','%%').encode('utf-8'))[2:].replace('\\','').replace('\'','').replace('[','').replace(']','').replace('[',''),1)

