###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from set_encoder.setenc_modules import *


class SetPool(nn.Module):
  def __init__(self, dim_input, num_outputs, dim_output,
        num_inds=32, dim_hidden=128, num_heads=4, ln=False, mode=None):
    super(SetPool, self).__init__()
    if 'sab' in mode: # [32, 400, 128]
      self.enc = nn.Sequential(
        SAB(dim_input, dim_hidden, num_heads, ln=ln),  # SAB?
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
    else: # [32, 400, 128]
      self.enc = nn.Sequential(
        ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),  # SAB?
        ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
    if 'PF' in mode: #[32, 1, 501]
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        nn.Linear(dim_hidden, dim_output))
    elif 'P' in mode:
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln))
    else: #torch.Size([32, 1, 501])
      self.dec = nn.Sequential(
        PMA(dim_hidden, num_heads, num_outputs, ln=ln), # 32 1 128
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        nn.Linear(dim_hidden, dim_output))
  # "", sm, sab, sabsm
  def forward(self, X):
    x1 = self.enc(X)
    x2 = self.dec(x1)
    return x2
