import torch.nn as nn
import torch

class LayerNorm(nn.Module):
  def __init__(self, embed_dim, eps=1e-12):
      super(LayerNorm, self).__init__()
      self.gamma = nn.Parameter(torch.ones(embed_dim))
      self.beta = nn.Parameter(torch.zeros(embed_dim))
      self.eps = eps

  def forward(self, x):
      mean = x.mean(-1, keepdim=True)
      var = x.var(-1, unbiased=False, keepdim=True)
      out = (x - mean) / torch.sqrt(var + self.eps)
      out = self.gamma * out + self.beta

      return out

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.layer1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p = dropout)
        self.layer2 = nn.Linear(projection_dim, projection_dim)
        self.norm = LayerNorm(projection_dim)

    
    def forward(self, x):

        out = self.layer1(x)
        out_skip = out.clone()
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = out + out_skip
        out = self.norm(out)
        return out