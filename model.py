import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

trim = lambda x : x[:, :(x.shape[1] - torch.min(torch.sum(x.eq(0).long(), dim=1))).item()]

def compute_mask(t, padding_idx=0):
    """
    compute mask on given tensor t
    :param t:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(t, padding_idx).float()
    return mask

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l
                         in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid,
                                   hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in
                         range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        nlayers = self.nlayers
        mask = compute_mask(input.transpose(1, 0))

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        lengths = mask.eq(1).long().sum(1)  # bs
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)  # bs
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # bs

        emb_sort = emb.index_select(1, idx_sort)  # sl * bs * ninp
        hid_sort = [(h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) for h in hidden]
        emb_sort = torch.nn.utils.rnn.pack_padded_sequence(emb_sort, lengths_sort)

        #         raw_output = emb_sort
        new_hidden = []
        raw_outputs = []
        raw_outputs_sorted = []
        outputs = []

        for l, rnn in enumerate(self.rnns):
            current_input = emb_sort
            emb_sort, new_h = rnn(emb_sort, hid_sort[l])
            emb_sort, _ = torch.nn.utils.rnn.pad_packed_sequence(emb_sort)

            new_hidden.append(new_h)
            raw_outputs.append(emb_sort)

            if l != nlayers - 1:
                emb_sort = lockdrop(emb_sort, dropouth)
                outputs.append(emb_sort)

        raw_outputs = [raw_output.index_select(1, idx_unsort) for raw_output in raw_outputs]
        new_hidden = [(h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) for h_sort in
                      new_hidden]

        hidden = new_hidden

        output = self.lockdrop(emb_sort, self.dropout)
        outputs.append(output)
        result = output.view(output.size(0) * output.size(1), output.size(2))

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
