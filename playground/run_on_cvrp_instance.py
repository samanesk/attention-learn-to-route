import os
import numpy as np
import torch
# 000
import sys

sys.path.append('/home/jupyter/attention-learn-to-route')

# set the current directory to the root path
os.chdir('/home/jupyter/attention-learn-to-route')

from torch.utils.data import DataLoader
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP

model, _ = load_model('pretrained/cvrp_100/')
torch.manual_seed(1234)
dataset = CVRP.make_dataset(size=100, num_samples=10)


# Need a dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=1000)

# Make var works for dicts
batch = next(iter(dataloader))

# Run the model
model.eval()
model.set_decode_type('greedy')
with torch.no_grad():
    length, log_p, pi = model(batch, return_pi=True)
tours = pi