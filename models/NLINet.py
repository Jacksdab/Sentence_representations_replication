from models.AWE import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import argparse
import time
import os

class NLINet(nn.Module):
    def __init__(self, encoder_name, vocab):
        super().__init__()
        # load in glove & freeze the weights
        self.device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.embeddings = nn.Embedding.from_pretrained(vocab.vectors.to(self.device), freeze=True)
        
        if encoder_name == 'Baseline':
            self.encoder = Baseline()
            self.classifier = Classifier(input_dim= 4 * 300)
        if encoder_name == 'uniLSTM':
            self.encoder = LSTMEncoder()
            self.classifier = Classifier(input_dim= 4 * 2048)
        if encoder_name == 'BLSTM':
            self.encoder = BLSTM()
            self.classifier = Classifier(input_dim= 4 * 2 * 2048)
        if encoder_name == 'BLSTMMax':
            self.encoder = BLSTMMax()
            self.classifier = Classifier(input_dim= 4 * 2 *2048)
    
    def forward(self, premise, hypothesis):
        # premise and hypothesis are tuples of (x, x_length)
        # print('model device is', self.device)
        # print('this should be a tuple: ', premise, 'and premise size should be B, length', premise[0].shape)
        premise, p_lengths = premise
        hypo, h_lengths = hypothesis
        # print(premise.device)
        # print(p_lengths.device)
        # print(hypo.device)
        # print(h_lengths.device)
        # retrieve embedding vectors 
        premises = self.embeddings(premise).to(self.device)
        hypthesis = self.embeddings(hypo).to(self.device)

        # encode and apply interaction steps
        u = self.encoder(premises, p_lengths)
        v = self.encoder(hypthesis, h_lengths)
        difference = torch.abs(u-v)
        multiplication = u * v

        features = torch.cat((u,v, difference, multiplication), dim = 1)
        output = self.classifier(features)
        
        return output