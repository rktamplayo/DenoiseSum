import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class LM(nn.Module):

  def __init__(self, word_size, 
               word_dim, hidden_dim):
    super(LM, self).__init__()
    self.word_size = word_size
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim

    # EMBEDDING
    self.word_embs = nn.Embedding(num_embeddings=word_size,
                                  embedding_dim=word_dim)

    # ENCODER
    self.fw_encoder = nn.LSTMCell(word_dim, hidden_dim)
    self.bw_encoder = nn.LSTMCell(word_dim, hidden_dim)

    # CLASSIFIER
    self.classifier = nn.Linear(hidden_dim*2, word_size)

    # OTHERS
    self.dropout = nn.Dropout(0.1)


  def forward(self, text, mask, training=True):
    batch_size, seq_len = text.size()

    # EMBED
    xs = self.word_embs(text)
    xs = self.dropout(xs)

    # ENCODE
    ht = ct = xs.new_zeros(batch_size, self.hidden_dim)
    fw_hs = []
    for i in range(seq_len):
      xt = xs[:,i]
      ht, ct = self.fw_encoder(xt, (ht, ct))
      fw_hs = fw_hs + [ht]
    fw_hs = torch.stack(fw_hs, dim=1)

    ht = ct = xs.new_zeros(batch_size, self.hidden_dim)
    bw_hs = []
    for i in reversed(range(seq_len)):
      xt = xs[:,i]
      ht, ct = self.bw_encoder(xt, (ht, ct))
      bw_hs = [ht] + bw_hs
    bw_hs = torch.stack(bw_hs, dim=1)

    hs = torch.cat([fw_hs[:,:-2], bw_hs[:,2:]], dim=-1)
    hs = self.dropout(hs)

    # CLASSIFY
    ps = self.classifier(hs)
    if not training:
      return ps

    # LOSS
    seq_len -= 2
    ys = text[:,1:-1].contiguous().view(batch_size*seq_len)
    mask = mask[:,1:-1].contiguous().view(batch_size*seq_len)

    batch_loss = F.cross_entropy(ps.view(batch_size*seq_len, -1), ys, reduction='none')
    batch_loss *= mask
    batch_loss = batch_loss.mean()

    return batch_loss