import itertools
import sys

import numpy as np, json




import sys, os
import requests
from io import BytesIO
import time
os.system('wget https://%s/%s/threads.json'%(sys.argv[2],sys.argv[3]))
time.sleep(1)
os.system(r"cat threads.json | egrep -o no\":[0-9]+ | sed s/no\"://g > threads.np")
tids=np.loadtxt('threads.np',dtype=np.int32)
os.system('rm threads.json')


for thread in tids:
    os.system('mkdir -p %s && wget -O %s/%i.json https://%s/%s/%s/%i.json'%(sys.argv[1],sys.argv[1],thread,sys.argv[2],sys.argv[3],sys.argv[4],thread))
    time.sleep(1)
time.sleep(10)
