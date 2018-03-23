import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NMT(nn.Module):
    def __init__(self, args, src_dict, dst_dict,  use_cuda=True, is_testing=False):
        super(NMT, self).__init__()
        self.args = args
        self.use_cuda = use_cuda
        self.is_testing = is_testing

        # Initialize encoder and decoder
        self.encoder = Encoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )
        self.decoder = AttnBasedDecoder(
            dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            use_cuda=use_cuda,
            is_testing=is_testing
        )

    def forward(self, sample):
        # encoder_output: (seq_len, batch, hidden_size * num_directions)
        # _encoder_hidden: (num_layers * num_directions, batch, hidden_size)
        # _encoder_cell: (num_layers * num_directions, batch, hidden_size)
        encoder_out = self.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
        
        # # The encoder hidden is  (layers*directions) x batch x dim.
        # # If it's bidirectional, We need to convert it to layers x batch x (directions*dim).
        # if self.args.bidirectional:
        #     encoder_hiddens = torch.cat([encoder_hiddens[0:encoder_hiddens.size(0):2], encoder_hiddens[1:encoder_hiddens.size(0):2]], 2)
        #     encoder_cells = torch.cat([encoder_cells[0:encoder_cells.size(0):2], encoder_cells[1:encoder_cells.size(0):2]], 2)

        decoder_out, predictions, attn_scores = self.decoder(sample['net_input']['prev_output_tokens'], encoder_out)

        return decoder_out, predictions

class Encoder(nn.Module):
    def __init__(self, dictionary, embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        # Define the LSTM encoder
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=dropout_out,
            bidirectional=False
            )

    def forward(self, src_tokens, src_tokens_len):

        bsz, seqlen = src_tokens.size()
        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T X C -> T X B X C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_tokens_len.data.tolist())

        # apply LSTM
        h0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        c0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x,
            (h0, c0),
        )

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, embed_dim]

        return x, final_hiddens, final_cells


class AttnBasedDecoder(nn.Module):
    def __init__(self, dictionary, encoder_embed_dim=512, embed_dim=512,
                 out_embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True, is_testing=False):
        super(AttnBasedDecoder, self).__init__()
        self.is_testing = is_testing
        self.use_cuda = use_cuda
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(encoder_embed_dim, embed_dim)
        if embed_dim != out_embed_dim:
            self.additional_fc = Linear(embed_dim, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)


    def forward(self, input_seq, encoder_out):

        bsz, seqlen = input_seq.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out
        srclen = encoder_outs.size(0)

        x = self.embed_tokens(input_seq) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        _, encoder_hiddens, encoder_cells = encoder_out
        num_layers = len(self.layers)
        prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
        prev_cells = [encoder_cells[i] for i in range(num_layers)]
        input_feed = Variable(x.data.new(bsz, embed_dim).zero_())
        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())

        if self.use_cuda:
            input_feed = input_feed.cuda()
            attn_scores = attn_scores.cuda()

        outs = []
        predictions = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if j == 0 or not self.is_testing:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = torch.cat((last_pred_word, input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # project back to size of vocabulary
            out = self.fc_out(out)
            out = F.log_softmax(out, dim=1)

            if not self.is_testing:
                outs.append(out)
            else:
                # get the word distribution and select the one with the
                # highest probability as the next input. Save the prediction result
                top_val, top_inx = out.topk(1)
                pred = top_inx.squeeze(1)
                predictions.append(pred)
                last_pred_word = self.embed_tokens(pred)


        # collect outputs across time steps
        if not self.is_testing:
            x = torch.cat(outs, dim=0).view(seqlen, bsz, -1)
            # T x B x C -> B x T x C
            x = x.transpose(1, 0)
        else:
            predictions = torch.cat(predictions, dim=0).view(bsz, seqlen)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)


        return x, predictions, attn_scores

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super(AttentionLayer, self).__init__()

        self.input_proj = nn.Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2 * output_embed_dim, output_embed_dim, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: bze x hsze
        # encoder_outputs: seq x bze x hsze

        # x: bze x hze
        x = self.input_proj(hidden)

        # compute attention
        attn_scores = (encoder_outputs * x.unsqueeze(0)).sum(dim=2)

        attn_weights = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        context = (attn_weights.unsqueeze(2) * encoder_outputs).sum(dim=0)

        attn_vector = F.tanh(self.output_proj(torch.cat((context, hidden), dim=1)))

        return attn_vector, attn_weights


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.uniform_(-0.1, 0.1)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m