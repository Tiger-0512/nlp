import prepare

word2index = prepare.word2index
make_wakati = prepare.make_wakati
sentence2index = prepare.sentence2index

VOCAB_SIZE = len(word2index)
EMBEDDING_DIM = 10
HIDDEN_DIM = 128
embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
s1 = "震災をうけて感じた、大切だと思ったこと"
print(make_wakati(s1))
#['震災', 'を', 'うけ', 'て', '感じ', 'た', '大切', 'だ', 'と', '思っ', 'た', 'こと']

inputs1 = sentence2index(s1)
emb1 = embeds(inputs1)
lstm_inputs1 = emb1.view(len(inputs1), 1, -1)
out1, out2 = lstm(lstm_inputs1)
print(out1)
print(out2)
# out1
#tensor([[[-0.0146, -0.0069,  0.0323,  ..., -0.0091, -0.0313,  0.0114]],
#        [[-0.0321, -0.0447,  0.0491,  ...,  0.0175, -0.0253,  0.0031]],
#        [[-0.0091, -0.0532,  0.0144,  ..., -0.0411, -0.0329, -0.0310]],
#        ...,
#        [[-0.0061,  0.0423,  0.0123,  ..., -0.0647, -0.0303, -0.0459]],
#        [[-0.0410,  0.0180,  0.0554,  ..., -0.0595, -0.0158, -0.0479]],
#        [[ 0.0323, -0.0564, -0.0181,  ...,  0.0236, -0.0057,  0.0101]]],
#       grad_fn=<StackBackward>)
# out2
#(tensor([[[ 0.0323, -0.0564, -0.0181,  0.0247, -0.0147,  0.0248,  0.0125,
#          (長いので省略)
#          -0.0057,  0.0101]]], grad_fn=<StackBackward>),
#          tensor([[[ 0.0711, -0.1137, -0.0448,  0.0477, -0.0253,  0.0564,  0.0251,
#          -0.1323,  0.1250,  0.0682,  0.0218, -0.0083, -0.0245,  0.0315,
#          (長いので省略)
#          -0.0124,  0.0266]]], grad_fn=<StackBackward>))