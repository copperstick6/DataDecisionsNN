import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from utils import load
from models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def test(iterations, batch_size=256):
    train_inputs, train_labels = load(os.path.join('datasets/test_data.npy'))

    model = ConvNetModel()
    pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

    model.load_state_dict(torch.load(os.path.join(dirname, 'deep')))
    model.eval()

    accuracies = []
    for iteration in range(iterations):
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        pred_val = torch.Tensor(pred(model(batch_inputs)))
        accuracies.append( np.mean(pred_val.data.numpy() == train_labels[batch].view(-1).data.numpy(), keepdims= True) )
    print(accuracies)
    print( 'Accuracy ', np.mean(accuracies), '+-', np.std(accuracies)/np.sqrt(len(accuracies)))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=10)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.iterations)
