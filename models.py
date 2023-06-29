import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# import pdb 


class VesselRouteForecasting(nn.Module):
    def __init__(self, rnn_cell=nn.LSTM, input_size=4, hidden_size=150, num_layers=1,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super(VesselRouteForecasting, self).__init__()
        
        # Input and Recurrent Cell
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.rnn_cell = rnn_cell(
            input_size=input_size, 
            num_layers=self.num_layers, 
            hidden_size=self.hidden_size, 
            batch_first=self.batch_first, 
            bidirectional=self.bidirectional, 
            **kwargs
        )

        fc_layer = lambda in_feats, out_feats: nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=out_feats),
            nn.ReLU(),
        )

        # Output layers
        fc_layers = [2 * hidden_size if self.bidirectional else hidden_size, *fc_layers, 2]
        self.fc = nn.Sequential(
            *[fc_layer(in_feats, out_feats) for in_feats, out_feats in zip(fc_layers, fc_layers[1:-1])],
            nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1])
        )
                        
        self.scale = scale
        if self.scale is not None:
            self.mu, self.sigma = self.scale['mu'], self.scale['sigma']    


    def forward_rnn_cell(self, x, lengths):
        # Sort input sequences by length in descending order
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        sorted_x = x[sorted_idx]

        # Pack the sorted sequences
        packed_x = pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True)

        # Initialize ```hidden state``` and ```cell state``` with zeros
        # h0, c0 = torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device),\
        #          torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate packed sequences through LSTM
        # packed_out, (h_n, c_n) = self.rnn_cell(packed_x, (h0, c0))
        packed_out, (h_n, c_n) = self.rnn_cell(packed_x)
        
        # Reorder the output sequences to match the original input order
        _, reversed_idx = sorted_idx.sort(0)

        return packed_out, (h_n, c_n), sorted_idx, reversed_idx
    

    def forward(self, x, lengths):
        # Initialize ```hidden state``` and ```cell state``` with zeros
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        packed_out, (h_n, c_n), _, ix = self.forward_rnn_cell(x, lengths)

        # Unpack the output sequences
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Decode the hidden state of the last time step
        out = self.fc(
            torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1]
        )     
        
        return torch.add(torch.mul(out[ix], self.sigma), self.mu)


class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform_(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = torch.ones(attentions.size()).to(attentions.device)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################
        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # pdb.set_trace()
        representations = weighted.sum(dim=1).squeeze()
        return representations, scores


class AttentionVRF(VesselRouteForecasting):
    def __init__(self, rnn_cell=nn.LSTM, input_size=4, hidden_size=150, num_layers=1,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super().__init__(
            rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=batch_first, fc_layers=fc_layers, scale=scale, bidirectional=bidirectional, **kwargs
        )
        
        # Attention (is really all you need?)
        attention_size = 2 * hidden_size if bidirectional else hidden_size
        self.attention, self.attention_scores = SelfAttention(attention_size, batch_first=batch_first), None


    def forward(self, x, lengths):
        # Initialize ```hidden state``` and ```cell state``` with zeros
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        packed_out, (h_n, c_n), ix_sort, ix_rev = self.forward_rnn_cell(x, lengths)

        # Unpack the output sequences
        padded_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        representations, self.attention_scores = self.attention(padded_out, lengths[ix_sort])

        # Decode the hidden state of the last time step
        out = self.fc(representations).view(x.shape[0], -1)

        return torch.add(torch.mul(out[ix_rev], self.sigma), self.mu)
