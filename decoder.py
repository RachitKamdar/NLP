import torch, math, copy
from torch import nn, Tensor
from torch.nn import Embedding, LayerNorm
from utils import clones, SublayerConnection, PositionwiseFFN

class DecoderBlock(torch.nn.Module):
  def __init__(self, size: int, 
              self_attn: MultiHeadedAttention, 
              encoder_attn: MultiHeadedAttention, 
              ffn: PositionwiseFFN, 
              dropout=0.1):
      super(DecoderBlock, self).__init__()
      self.size = size
      self.self_attn = self_attn
      self.encoder_attn = encoder_attn
      self.ffn = ffn
      self.sublayers = clones(SublayerConnection(size, dropout), 3)
  def forward(self, x, encoder_states, source_mask, target_mask):
    x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x,
    target_mask))
    x = self.sublayers[1](x, lambda x: self.encoder_attn(x,
    encoder_states, encoder_states, source_mask))
    return self.sublayers[2](x, self.ffn)


class Decoder(torch.nn.Module):
  def __init__(self, block: DecoderBlock, N: int, vocab_size: int):
    super(Decoder, self).__init__()
    self.blocks = clones(block, N)
    self.norm = LayerNorm(block.size)
    self.projection = torch.nn.Linear(block.size, vocab_size)
  def forward(self, x, encoder_states, source_mask, target_mask):
    for layer in self.blocks:
      x = layer(x, encoder_states, source_mask, target_mask)
      x = self.norm(x)
    return torch.nn.functional.log_softmax(self.projection(x), dim=-1)