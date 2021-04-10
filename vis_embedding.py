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
    indices.append(word[0])
    print(word[1])
X_embedded=TSNE(n_components=2, verbose=2).fit_transform(e[indices,:])
df = pd.DataFrame(columns=['x', 'y', 'word'])
df['x'], df['y'], df['word'] = X_embedded[:,0], X_embedded[:,1], [vocab[i] for i in indices]

source = ColumnDataSource(ColumnDataSource.from_df(df))
labels = LabelSet(x="x", y="y", text="word", y_offset=8,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')

plot = figure(plot_width=600, plot_height=600)
plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
plot.add_layout(labels)
show(plot)
pca_result = pca.fit_transform(e)
