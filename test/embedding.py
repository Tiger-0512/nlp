import get_data
import torch
import torch.nn as nn

# 以下の宣言で行が単語ベクトル、列が単語のインデックスのマトリクスを生成してる感じ
embeds = nn.Embedding(10, 6) # (Embedding(単語の合計数, ベクトル次元数))

# ３行目の要素を取り出したいならば
w1 = torch.tensor([2])
print(embeds(w1))
# tensor([[-1.5947, -0.8387,  0.7669, -0.9644, -0.7902,  2.7167]],
#        grad_fn=<EmbeddingBackward>)

# 3行目、5行目、１０行目の要素を取り出したいならば、
w2 = torch.tensor([2,4,9])
print(embeds(w2))
# tensor([[-1.5947, -0.8387,  0.7669, -0.9644, -0.7902,  2.7167],
#        [ 0.0405,  1.4236,  0.1947,  0.2609,  0.2047, -1.4964],
#        [ 1.7325, -0.2543, -0.5139, -0.9527, -0.1344,  0.0984]],
#       grad_fn=<EmbeddingBackward>)