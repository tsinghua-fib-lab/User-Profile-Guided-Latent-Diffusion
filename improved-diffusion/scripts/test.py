import numpy as np
import torch
import pdb
import pickle

fct = torch.nn.CrossEntropyLoss(reduction='none')
ts = pickle.load(open('/data2/songyiwen/human_traj_diffusion/improved-diffusion/scripts/token_loss_tensors.pkl','rb'))
logits = torch.from_numpy(ts['logits'])
input_ids = torch.from_numpy(ts['input_ids'])
decoder_nll = fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)

# print(decoder_nll.shape)
decoder_nll = decoder_nll.mean(dim=-1)
pdb.set_trace()