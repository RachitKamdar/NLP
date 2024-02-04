import torch, math, copy

def clones(module: torch.nn.Module, n: int):
  return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionwiseFFN(torch.nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout=0.1):
    super(PositionwiseFFN, self).__init__()
    self.w_1 = torch.nn.Linear(d_model, d_ff)
    self.w_2 = torch.nn.Linear(d_ff, d_model)
    self.dropout = torch.nn.Dropout(dropout)
  def forward(self, x):
    return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SublayerConnection(torch.nn.Module):
  def __init__(self, size, dropout):
    super(SublayerConnection, self).__init__()
    self.norm = LayerNorm(size)
    self.dropout = torch.nn.Dropout(dropout)
  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))