from torch import nn, Tensor
from torch.nn import Embedding
from utils import clones

def attention(query, key, value, mask=None, dropout=None):
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = torch.nn.functional.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    self.d_k = d_model // h
    self.h = h
    self.fc_layers = clones(torch.nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = torch.nn.Dropout(p=dropout)
  def forward(self, query, key, value, mask=None):
    if mask is not None:
      mask = mask.unsqueeze(1)
    batch_samples = query.size(0)
    projections = list()
    for l, x in zip(self.fc_layers, (query, key, value)):
      projections.append(l(x).view(batch_samples, -1, self.h, self.d_k).transpose(1, 2))
    query, key, value = projections
    x, self.attn = attention(query, key, value,mask=mask,dropout=self.dropout)
    x = x.transpose(1, 2).contiguous().view(batch_samples, -1, self.h * self.d_k)
    return self.fc_layers[-1](x)