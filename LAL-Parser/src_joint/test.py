import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
#import src_dep_const_test.chart_helper as chart_helper
import dcchart_helper
import torch

loss = torch.FloatTensor(1).fill_(0.)
print(loss)
print(loss.data)
loss_value = float(loss.data.cpu().numpy())
