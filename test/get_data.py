import os
from glob import glob
import pandas as pd
import linecache

# カテゴリを配列で取得
categories = [name for name in os.listdir("/Users/tiger/repos/dataset/text/") if os.path.isdir("/Users/tiger/repos/dataset/text/" + name)]
print(categories)
# ['movie-enter', 'it-life-hack', 'kaden-channel', 'topic-news', 'livedoor-homme', 'peachy', 'sports-watch', 'dokujo-tsushin', 'smax']

datasets = pd.DataFrame(columns=["title", "category"])
for cat in categories:
    path = "/Users/tiger/repos/dataset/text/" + cat + "/*.txt"
    files = glob(path)
    for text_name in files:
        title = linecache.getline(text_name, 3)
        s = pd.Series([title, cat], index=datasets.columns)
        datasets = datasets.append(s, ignore_index=True)

# データフレームシャッフル
datasets = datasets.sample(frac=1).reset_index(drop=True)
datasets.head()
#title  category
#0  兼用アンテナ搭載の「Viewer Dock」が同梱！シャープのドコモ向けハイエンドエンタメ系... smax
#1  女は“愛嬌”、男も“愛嬌”-人事担当者がこっそり教える採用ウラ話 vol.6\n  livedoor-homme
#2  社会貢献×ファッションがカッコイイ、今年の春旋風を巻き起こしたMODE for Charit...  peachy
#3  今でも、後でも読めるニュースがここにある！スマホでもタブレットでも読みやすいITニュース活用...   it-life-hack
#4  被災地の缶詰を途上国に…「正気じゃない。人殺しだ!!」\n topic-news
