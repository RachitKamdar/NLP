import torch, math, copy
from torch import nn, Tensor
from torch.nn import Embedding, LayerNorm
from utils import clones, SublayerConnection, PositionwiseFFN

class EncoderBlock(torch.nn.Module):
  def __init__(self,
  size: int,
  self_attn: MultiHeadedAttention,
  ffn: PositionwiseFFN,
  dropout=0.1):
    super(EncoderBlock, self).__init__()
    self.self_attn = self_attn
    self.ffn = ffn
    self.sublayers = clones(SublayerConnection(size, dropout), 2)
    self.size = size
  def forward(self, x, mask):
    x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
    return self.sublayers[1](x, self.ffn)

class Encoder(torch.nn.Module):
  def __init__(self, block: EncoderBlock, N: int):
    super(Encoder, self).__init__()
    self.blocks = clones(block, N)
    self.norm = LayerNorm(block.size)
  def forward(self, x, mask):
    for layer in self.blocks:
      x = layer(x, mask)
    return self.norm(x)