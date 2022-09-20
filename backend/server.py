import numpy as np
import os
from flask import Flask, request, render_template, redirect
app = Flask(__name__)
import torch
from torch import nn, optim
import json

from models import *

ratings = ['C', 'CC', 'CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']


@app.route("/", methods =['GET'])
def routeFrontend():
    return render_template('frontend.html')

@app.route("/api", methods=['POST'])
def evaluateModel():
    data = request.form.get("json")
    data = json.loads(data)
    data[0] = [ 0 if x == '' else x for x in data[0]]
    data[0] = [float(x) for x in data[0]]
    model = ConvNetModel()
    pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
    model.load_state_dict(torch.load('deep'))
    model.eval()
    batch_inputs = torch.Tensor(np.array([data][0]))
    pred_val = torch.Tensor(pred(model(batch_inputs)))
    return ratings[int(pred_val.data.numpy().tolist()[0])]


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
