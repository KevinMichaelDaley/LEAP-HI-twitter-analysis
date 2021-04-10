import praw, time, sys
CID=open('rcid').read().strip()
CSECRET=open('rcsecret').read().strip()
UAGENT=open('ruagent').read().strip()
reddit = praw.Reddit(client_id=CID, client_secret=CSECRET, user_agent=UAGENT)
hot_posts = reddit.subreddit(sys.argv[1]).new(limit=10)
for submission in hot_posts:
    print(submission.author, " ", submission.title, " ", submission.url, submission.body if hasattr(submission, 'body') else '')
    submission.comment_sort='new'
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        print('\t', comment.author, " ",  comment.body)
        sys.stdout.flush()
time.sleep(4)
