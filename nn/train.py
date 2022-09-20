import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from utils import load
from models import *

dirname = os.path.dirname(os.path.abspath(__file__))


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    return np.mean(outputs_idx.data.numpy() == labels.view(-1).data.numpy(), keepdims=True)

def train(iterations, batch_size=256, log_dir=None):
    '''
    This is the main training function, feel free to modify some of the code, but it
    should not be required to complete the assignment.
    '''

    """
    Load the training data
    """
    model=ConvNetModel()
    inputs, labels = load(os.path.join('datasets/train_data.npy'))

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)

    running_loss = 0.0
    print(iterations)

    for iteration in range(iterations):
        batch = np.random.choice(inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(inputs[batch], dtype=torch.float32)
        batch_labels = torch.as_tensor(labels[batch], dtype=torch.long)

        # Set the state of the model to 'train'. This is important for backpropagation step about which you
        # learn in the next class. Backpropagation requires the values computed in the forward() function for
        # each layer. Invoking the .train() function directs the model to save these values for future
        # backpropagation step.
        model.train()

        # Very important to reset the gradients to zero before training a minibatch. Otherwise it will keep
        # adding to gradients computed in the previous iterations
        optimizer.zero_grad()


        outputs = model(batch_inputs)
        t_loss_val = loss(outputs, batch_labels.squeeze())
        t_acc_val = accuracy(outputs, batch_labels)

        # Perform the backpropagation step to compute gradients
        t_loss_val.backward()
        # Perform the gradient update
        optimizer.step()

        if iteration % 10 == 0:
            model.eval()

            print('[%5d]'%iteration, 'loss = %f'%t_loss_val, 'acc = %f'%t_acc_val)

    # Save the trained model
    dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
    torch.save(model.state_dict(), os.path.join(dirname, 'deep')) # Do NOT modify this line

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=200)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    print ('[I] Start training')
    train(args.iterations, log_dir=args.log_dir)
    print ('[I] Training finished')
