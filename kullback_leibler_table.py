# Build the kullback liebler tabels
import numpy as np
import matplotlib.path as mpltPath
import xml.etree.ElementTree as ET
from tweat.stream import MySQLInterface as sql
from collections import defaultdict


def gen_state_polys():
    tree = ET.parse('states.xml')
    root = tree.getroot()
    polys = {}
    for child in root:
        if child.tag != 'state':
            continue
        plist = []
        for p in child:
            if p.tag != 'point':
                continue
            plist.append([float(p.attrib['lng']), float(p.attrib['lat'])])
        polys[child.attrib['name'].lower().strip()] = np.array(plist)
    return polys


def state_has_p(state_bounds, lnglat):
    path = mpltPath.Path(state_bounds)
    return path.contains_points(lnglat)


def longlat2state(all_state_bounds, lnglat):
    for s in all_state_bounds:
        if state_has_p(all_state_bounds[s], np.array([lnglat])):
            return s
    return None


def build_dictionaries():
    filters = open('labeled_sips_combined.txt', 'r')
    dangerous_world = []
    fear_of_regulation = []
    for f in filters:
        if f.strip() == "":
            continue
        ww = f.strip().split()
        if int(ww[3]) == 1:
            dangerous_world.append(ww[0])
        if int(ww[3]) == 2:
            fear_of_regulation.append(ww[0])

    return {k:0 for k in dangerous_world}, {k:0 for k in fear_of_regulation}


def build_model():
    model = {}
    states = gen_state_polys()
    dangerous_world, fear_of_regulation = build_dictionaries()
    for state in states:
        model[state] = {"dangerous_word": dangerous_world.copy(),
                        "fear_of_regulation": fear_of_regulation.copy()}
    return model


def main(file, database, username, password):
    db = sql('localhost', username, password, database)
    alls = gen_state_polys()
    A = np.loadtxt(file)
    model = build_model()
    for a in A:
        id = int(a[-3])
        lnglat = [a[2], a[1]]
        s = longlat2state(alls, lnglat)
        if s is None:
            continue
        state_model = model[s]
        tweet = db.query('select TEXT from TWEETS where ID=%i limit 1' % id)[0][0]
        for bag_of_words in state_model.values():
            for word in tweet.split():
                word = word.lower()
                if word in bag_of_words.keys():
                    bag_of_words[word] = bag_of_words[word] + 1
    for state in model:
        state_model = model[state]
        danger = state_model['dangerous_word'].values()
        fear = state_model['fear_of_regulation'].values()
        total = sum(list(danger) + list(fear))
        print(f"{state}, {sum(danger) / total},  {sum(fear) / total}")

if __name__ == '__main__':
    main('./all_tweets.np', 'gun', 'kslote', 'password')


