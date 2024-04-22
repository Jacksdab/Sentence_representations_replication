import torch
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 3)
        )

    def forward(self, sent_embeddings):
        #sent_embeddings : [B, 4 * embed_dim] -> this differes per model. 
        # 4 = concat (2) + mult + subtr 
        preds = self.classifier(sent_embeddings)
        return preds

class Baseline(nn.Module):
    def  __init__(self) -> None:
        # (batch, sentence_length, embedding_dim)
        super().__init__()
    def forward(self, x, x_len):
        # take the mean over sentence_length
        x = torch.mean(x, 1)
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_size = 2048):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.encoder_dim = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x, x_len):
        # x: (batch, sent_lengths, embedding_dim)

        # initialisation of hidden and cell state 
        batch_size = x.shape[0]
        self.hidden = torch.zeros((1, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)
        self.cell = torch.zeros((1, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)

        # sort by sentence lengths, keep indices for unsort for packing || remember batch comes first
        sorted_lengths, sorted_idx = torch.sort(x_len, descending= True, dim =0)
        sorted_sent = torch.index_select(x, dim=0 ,index = sorted_idx)

        # padding   
        sent_packed = nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_lengths.cpu(), batch_first=True)

        # obtain hidden state and remove the singleton dimension
        hidden_states = self.lstm(sent_packed, (self.hidden, self.cell))[1][0].squeeze(0)

        # unsort to obtain original order
        unsort_idx = torch.argsort(sorted_idx)
        sent_representations = torch.index_select(hidden_states, dim=0, index=unsort_idx)

        return sent_representations
    

class BLSTM(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_size = 2048):
        super().__init__()

        # Define the LSTM parameters
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.encoder_dim = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, x_len):
        
        # Set initial states
        batch_size = x.shape[0]
        self.hidden = torch.zeros((2, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)
        self.cell = torch.zeros((2, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)

        # sort by sentence lengths, keep indices for unsort for packing || remember batch comes first
        sorted_lengths, sorted_idx = torch.sort(x_len, descending= True, dim =0)
        sorted_sent = torch.index_select(x, dim=0 ,index = sorted_idx)

        # padding   
        sent_packed = nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_lengths.cpu(), batch_first=True)

        # obtain hidden state and remove the singleton dimension
        hidden_states = self.lstm(sent_packed, (self.hidden, self.cell))[1][0].squeeze(0)

        # Extract the last hidden states of the forward and backward LSTMs
        forward_hidden = hidden_states[0]
        backward_hidden = hidden_states[1]


        # Ensure the hidden states are of correct dimensions
        if forward_hidden.dim() > 2:
            forward_hidden = forward_hidden.squeeze(0)
            backward_hidden = backward_hidden.squeeze(0)

        # Concatenate the forward and backward hidden states
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Unsort the hidden states to original order
        unsorted_indices = torch.argsort(sorted_idx)
        final_embeddings = combined_hidden.index_select(dim=0, index=unsorted_indices)

        return final_embeddings

class BLSTMMax(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_size = 2048):
        super().__init__()

        # Define the LSTM parameters
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.encoder_dim = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, x_len):
        
        # Set initial states
        batch_size = x.shape[0]
        self.hidden = torch.zeros((2, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)
        self.cell = torch.zeros((2, batch_size, self.encoder_dim), dtype=torch.float, device=self.device)

        # sort by sentence lengths, keep indices for unsort for packing || remember batch comes first
        sorted_lengths, sorted_idx = torch.sort(x_len, descending= True, dim =0)
        sorted_sent = torch.index_select(x, dim=0 ,index = sorted_idx)

        # padding   
        sent_packed = nn.utils.rnn.pack_padded_sequence(sorted_sent, sorted_lengths.cpu(), batch_first=True)

        # obtain hidden state and remove the singleton dimension
        output, _ = self.lstm(sent_packed, (self.hidden, self.cell))

        # pad output
        hidden_states = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


        # Unsort the hidden states to original order
        unsorted_indices = torch.argsort(sorted_idx)
        final_embeddings = hidden_states[0].index_select(dim=0, index=unsorted_indices)

        # max pooling
        sent_output = [x[:l] for x, l in zip(final_embeddings, x_len)]
        final_embeddings = [torch.max(x, 0)[0] for x in sent_output]
        final_embeddings = torch.stack(final_embeddings, 0)

        return final_embeddings

