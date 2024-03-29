import os.path
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import json


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, query, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(query, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, query, k, v, mask=None):

        bs = query.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        query = self.q_linear(query).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        query = query.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(query, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position=-1):
        if position == -1:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, position:position+1]
        return self.dropout(x)


class BeliefTracker(nn.Module):
    def __init__(self, args, num_labels, device):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.device = device
        self.lamb = args.lamb
        self.args = args

        # Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(args.bert_dir, 'bert-base-uncased.model')
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob

        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        # slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
                os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])
        self.mix_value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        # Attention layer
        # attention for slot and turn-level context
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0.)
        # attention for slot and passage-level context
        self.attn2 = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0.)
        # attention for slot and previous dialogue state context
        if self.args.utterance_level_combine:
            self.attn1 = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0.)
        # attention for passage dialogue and dialogue state context
        if self.args.passage_level_combine:
            self.attn3 = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0.)

        self.add_pe = PositionalEncoding(self.bert_output_dim, 0.)

        # Belief Tracker
        self.nbt = Encoder(EncoderLayer(
                            self.bert_output_dim,
                            MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=self.hidden_dropout_prob),
                            PositionwiseFeedForward(self.bert_output_dim, self.hidden_dim, self.hidden_dropout_prob),
                            self.hidden_dropout_prob
                            ),
                            N=6
                           )

        self.linear = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        self.sent_layer_norm = nn.LayerNorm(self.bert_output_dim)
        self.word_layer_norm = nn.LayerNorm(self.bert_output_dim)
        self.query_sent_layer_norm = nn.LayerNorm(self.bert_output_dim)
        self.query_word_layer_norm = nn.LayerNorm(self.bert_output_dim)
        self.gate = nn.Linear(self.bert_output_dim * 2, self.bert_output_dim)
        self.clsf_update = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        self.update_linear = nn.Linear(self.bert_output_dim*2, 1)
        if self.args.utterance_level_combine:
            self.slot_history_gate = nn.Linear(self.bert_output_dim*2, 1)
        if self.args.passage_level_combine:
            self.passage_history_gate = nn.Linear(self.bert_output_dim * 2, 1)

        # Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        # Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        # Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids, combine_sv_encode=0):
        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(self.device)
        slot_mask = slot_ids > 0

        hid_slot, _ = self.sv_encoder(slot_ids.view(-1, self.max_label_length),
                                   slot_type_ids.view(-1, self.max_label_length),
                                   slot_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        # Slot value encoding
        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                           label_type_ids.view(-1, self.max_label_length),
                                           label_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        # Slot value combine encoding
        max_sv_len = 0
        if combine_sv_encode:
            for s_id, label_id in enumerate(label_ids):
                for s_label in label_id.tolist():
                    value_len = len([x for x in s_label if x != 0])
                    slot_len = len([x for x in slot_ids[s_id].tolist() if x != 0])
                    combine_len = value_len + slot_len
                    if max_sv_len < combine_len:
                        max_sv_len = combine_len
            mix_label_ids = []
            for s_id, label_id in enumerate(label_ids):
                tmp_label_id = torch.zeros([label_id.shape[0], max_sv_len], dtype=torch.long).to(self.device)
                for elemnet_id, s_label in enumerate(label_id.tolist()):
                    value_element = [x for x in s_label if x != 0]
                    slot_element = [x for x in slot_ids[s_id].tolist() if x != 0]
                    combine_element = slot_element[0:-1] + value_element
                    tmp_label_id[elemnet_id][0:len(combine_element)] = torch.tensor(combine_element, dtype=torch.long).to(self.device)
                mix_label_ids.append(tmp_label_id)

            for s, mix_label_id in enumerate(mix_label_ids):
                mix_label_type_ids = torch.zeros(mix_label_id.size(), dtype=torch.long).to(self.device)
                mix_label_mask = mix_label_id > 0
                mix_hid_label, _ = self.sv_encoder(mix_label_id.view(-1, max_sv_len),
                                               mix_label_type_ids.view(-1, max_sv_len),
                                               mix_label_mask.view(-1, max_sv_len),
                                               output_all_encoded_layers=False)
                mix_hid_label = mix_hid_label[:, 0, :]
                mix_hid_label = mix_hid_label.detach()
                self.mix_value_lookup[s] = nn.Embedding.from_pretrained(mix_hid_label, freeze=True)
                self.mix_value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0: # padding
                    break
                elif len[i, j, 1] > 0: # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids,
                input_len,
                labels,
                update=None,
                n_gpu=1,
                target_slot=None,
                mt=True,
                prev_label_ids=None,
                no_teaching_force=False,
                combine_sv_encode=0):

        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0) # dialog size
        ts = input_ids.size(1) # turn size
        bs = ds*ts
        seq_length = input_ids.size(2)
        slot_dim = len(target_slot)
        prev_label_ids_len = 0

        if prev_label_ids is not None:
            prev_label_ids_len = prev_label_ids.shape[2]

        # Utterance encoding
        token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)
        # Select target slot embedding
        hid_slot = self.slot_lookup.weight[target_slot, :]

        out_slot_gate = []
        if no_teaching_force:
            dialogue_num = input_ids.shape[0]
            turn_num = input_ids[0].shape[0]
            dialogue_label_lists = []
            loss_slot = [0]*slot_dim
            update_hidden_general = torch.zeros(dialogue_num*slot_dim, turn_num, self.bert_output_dim).to(self.device)
            main_loss = torch.Tensor([0.0]).to(self.device)
            for dialogue_index in range(dialogue_num):
                dialogue_hidden, _ = self.utterance_encoder(
                    input_ids[dialogue_index],
                    token_type_ids[dialogue_index],
                    attention_mask[dialogue_index],
                    output_all_encoded_layers=False
                )
                hidden = torch.mul(dialogue_hidden, attention_mask[dialogue_index].view(-1, seq_length, 1).expand(dialogue_hidden.size()).float())
                hidden_repeat = hidden.repeat(slot_dim, 1, 1)
                ts = hidden.shape[0]
                padding_utter = (input_len[dialogue_index].sum(-1) != 0)
                turn_mask = padding_utter.unsqueeze(0).repeat(ts, 1) & subsequent_mask(ts).to(self.device)
                hid_slot_repeat = hid_slot.repeat(1, ts).view(ts*slot_dim, -1)

                # slot attention with turn-level context
                hidden_repeat = self.attn.forward(self.query_word_layer_norm(hid_slot_repeat),
                                          hidden_repeat,
                                          hidden_repeat,
                                          mask=attention_mask[dialogue_index].view(-1, 1, seq_length).repeat(slot_dim, 1, 1))

                sentence_level_curr_hidden = torch.zeros(slot_dim, ts, hidden.shape[-1]).to(self.device)
                sentence_level_hidden = torch.zeros(slot_dim, ts, hidden.shape[-1]).to(self.device)
                dialogue_label_list = []
                for turn_index in range(turn_num):
                    if turn_index == 0:
                        tmp_prev_label_ids = prev_label_ids[dialogue_index][0]
                        tmp_prev_label_ids = tmp_prev_label_ids.tolist()
                    else:
                        tmp_prev_label_ids = pred_slot

                    turn_prev_label_hidden = torch.zeros(1, len(tmp_prev_label_ids), self.value_lookup[0].weight[0].shape[-1]).to(self.device)
                    for general_slot_id, general_slot_value in enumerate(tmp_prev_label_ids):
                        if combine_sv_encode:
                            turn_prev_label_hidden[0][general_slot_id] = self.mix_value_lookup[general_slot_id].weight[general_slot_value]
                        else:
                            turn_prev_label_hidden[0][general_slot_id] = self.value_lookup[general_slot_id].weight[general_slot_value]
                    turn_prev_label_hidden = turn_prev_label_hidden.repeat(slot_dim, 1, 1)

                    # slot attention with previous belief state
                    if self.args.utterance_level_combine:
                        turn_prev_label_hidden_turn = self.attn1.forward(self.query_word_layer_norm(hid_slot),
                                                                    turn_prev_label_hidden,
                                                                    turn_prev_label_hidden)

                        slot_history_gate = torch.sigmoid(self.slot_history_gate(torch.cat([hidden_repeat[turn_index::ts], turn_prev_label_hidden_turn], -1)))

                        # out_slot_gate.append([x[0][0] for x in slot_history_gate.tolist()])
                        # print(turn_index)
                        # turn_level vector
                        turn_hidden_curr = (1 - slot_history_gate) * hidden_repeat[turn_index::ts] + slot_history_gate * turn_prev_label_hidden_turn
                        sentence_level_curr_hidden[:, turn_index:turn_index+1, :] = turn_hidden_curr
                    else:
                        sentence_level_curr_hidden[:, turn_index:turn_index + 1, :] = hidden_repeat[turn_index::ts]
                        turn_hidden_curr = hidden_repeat[turn_index::ts]

                    turn_hidden = self.word_layer_norm(turn_hidden_curr)
                    # add position vector
                    turn_hidden = self.add_pe.forward(turn_hidden, turn_index)
                    sentence_level_hidden[:, turn_index:turn_index+1, :] = turn_hidden
                    # gather history turn_level vector
                    tmp_hidden = sentence_level_hidden[:, 0:turn_index+1, :]
                    # gather history mask
                    tmp_turn_mask = turn_mask[:, 0:turn_index+1, 0:turn_index+1]
                    tmp_turn_mask_repeat = tmp_turn_mask.repeat(slot_dim, 1, 1)
                    tmp_hidden = self.nbt.forward(tmp_hidden, tmp_turn_mask_repeat)
                    tmp_hidden = self.attn2.forward(self.query_sent_layer_norm(hid_slot), tmp_hidden, tmp_hidden,
                                                    mask=tmp_turn_mask_repeat)
                    sigmoid = torch.sigmoid(self.gate(torch.cat([sentence_level_curr_hidden[:, turn_index:turn_index+1, :], tmp_hidden[:, turn_index:turn_index+1]], -1)))
                    tmp_hidden = sigmoid * sentence_level_curr_hidden[:, turn_index:turn_index + 1, :] + (1 - sigmoid) * tmp_hidden[:, turn_index:turn_index+1]
                    # passage_level merge
                    if self.args.passage_level_combine:
                        prev_label_hidden_passage = self.attn3.forward(tmp_hidden.view(-1, self.bert_output_dim),
                                                                       turn_prev_label_hidden, turn_prev_label_hidden)
                        prev_label_hidden_passage = prev_label_hidden_passage.view(-1, 1, self.bert_output_dim)
                        passage_history_gate = torch.sigmoid(
                            self.passage_history_gate(torch.cat([tmp_hidden, prev_label_hidden_passage], -1)))
                        tmp_hidden = (1 - passage_history_gate) * tmp_hidden + passage_history_gate * prev_label_hidden_passage
                    update_hidden_general[dialogue_index::dialogue_num, turn_index:turn_index+1] = tmp_hidden
                    tmp_hidden = self.sent_layer_norm(self.linear(self.dropout(tmp_hidden)))
                    pred_slot = []
                    for s, slot_id in enumerate(target_slot):  ## note: target_slot is successive
                        # loss calculation
                        hid_label = self.value_lookup[slot_id].weight
                        num_slot_labels = hid_label.size(0)
                        _hid_label = hid_label.unsqueeze(0).repeat(1, 1, 1).view(num_slot_labels, -1)
                        _hidden = tmp_hidden[s, :, :].unsqueeze(2).repeat(1, num_slot_labels, 1).view(num_slot_labels, -1)
                        _dist = self.metric(_hid_label, _hidden).view(1, 1, num_slot_labels)
                        if self.distance_metric == "euclidean":
                            _dist = -_dist
                        if labels is not None:
                            _loss = self.nll(_dist.view(1, -1), labels[dialogue_index, turn_index, s].view(-1))
                            _loss = _loss/float(turn_num*dialogue_num)
                            loss_slot[s] += _loss.item()
                            main_loss += _loss
                        _, pred = torch.max(_dist, -1)
                        pred_slot.append(pred[0][0].tolist())
                    dialogue_label_list.append(pred_slot)
                dialogue_label_lists.append(dialogue_label_list)

            loss_update = torch.Tensor([0.0]).to(self.device)
            acc_update = torch.Tensor([0.0]).to(self.device)
            if update is not None:
                # tanh activation
                hidden_update = torch.tanh(
                    self.clsf_update(F.dropout(update_hidden_general, p=self.args.mt_drop, training=self.training)))
                prob_update = torch.sigmoid(self.update_linear(torch.cat([hidden_update, torch.cat(
                    [torch.zeros_like(hidden_update[:, :1]).cuda(), hidden_update[:, :-1]], 1)], -1))).squeeze()
                prob_update = prob_update.view(slot_dim, ds, ts).permute(1, 2, 0).contiguous()
                mask = (update > -1).float()
                update_1 = update.masked_fill((1 - mask).byte(), 1)
                prob_update_1 = prob_update.masked_fill((1 - mask).byte(), 1)
                loss_update = F.binary_cross_entropy(prob_update_1, update_1, reduction="none")
                loss_update = loss_update.sum() / mask.sum()
                tp = torch.sum((prob_update >= 0.5) & (update == 1))
                tn = torch.sum((prob_update < 0.5) & (update == 0))
                acc_update = (tp + tn) / mask.sum()

            if mt is True and update is not None:
                loss = self.lamb * main_loss + (1 - self.lamb) * loss_update
            else:
                loss = main_loss
            dialogue_label_lists = torch.tensor(dialogue_label_lists, dtype=torch.long).to(self.device)
            accuracy = (dialogue_label_lists == labels).view(-1, slot_dim)
            acc_slot = torch.sum(accuracy, 0).float() / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
            acc = sum(torch.sum(accuracy, 1) / slot_dim).float() / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy
            return loss, loss_slot, acc, acc_slot, dialogue_label_lists, (main_loss, loss_update, acc_update)
        else:
            hidden, _ = self.utterance_encoder(input_ids.view(-1, seq_length),
                                               token_type_ids.view(-1, seq_length),
                                               attention_mask.view(-1, seq_length),
                                               output_all_encoded_layers=False)

            hidden = torch.mul(hidden, attention_mask.view(-1, seq_length, 1).expand(hidden.size()).float())
            hidden = hidden.repeat(slot_dim, 1, 1)  # [(slot_dim*ds*ts), bert_seq, hid_size]
            hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [(slot_dim*ds*ts), bert_seq, hid_size]
            prev_label_hidden = None

            # slot attention with context token level
            hidden = self.attn.forward(self.query_word_layer_norm(hid_slot), hidden, hidden,
                                       mask=attention_mask.view(-1, 1, seq_length).repeat(slot_dim, 1, 1))

            # get previous dialogue state
            if self.args.utterance_level_combine or self.args.passage_level_combine:
                reformed_prev_label_ids = prev_label_ids.view(-1, prev_label_ids_len)
                prev_label_hidden = torch.zeros(reformed_prev_label_ids.shape[0], reformed_prev_label_ids.shape[1], self.value_lookup[0].weight[0].shape[-1]).to(self.device)
                for general_turn in range(reformed_prev_label_ids.shape[0]):
                    for general_slot_id in range(reformed_prev_label_ids.shape[1]):
                        if combine_sv_encode:
                            prev_label_hidden[general_turn][general_slot_id] = self.mix_value_lookup[general_slot_id].weight[reformed_prev_label_ids[general_turn][general_slot_id]]
                        else:
                            prev_label_hidden[general_turn][general_slot_id] = self.value_lookup[general_slot_id].weight[reformed_prev_label_ids[general_turn][general_slot_id]]
                prev_label_hidden = prev_label_hidden.repeat(slot_dim, 1, 1)

            # slot attention with previous dialogue state token level
            # merge the information in the slot attention with context token_level
            if self.args.utterance_level_combine:
                prev_label_hidden_turn = self.attn1.forward(self.query_word_layer_norm(hid_slot), prev_label_hidden, prev_label_hidden)
                slot_history_gate = torch.sigmoid(self.slot_history_gate(torch.cat([hidden, prev_label_hidden_turn], -1)))
                hidden = (1 - slot_history_gate)*hidden + slot_history_gate*prev_label_hidden_turn

            hidden = hidden.squeeze() # [slot_dim*ds*ts, bert_dim]
            hidden_curr = hidden.view(slot_dim, ds, ts, -1).view(-1, ts, self.bert_output_dim)
            hidden = self.word_layer_norm(hidden_curr)
            hidden = self.add_pe.forward(hidden)

            # NBT
            turn_mask = torch.Tensor(ds, ts, ts).byte().to(self.device)
            for d in range(ds):
                padding_utter = (input_len[d, :].sum(-1) != 0)
                turn_mask[d] = padding_utter.unsqueeze(0).repeat(ts, 1) & subsequent_mask(ts).to(self.device)

            turn_mask = turn_mask.repeat(slot_dim, 1, 1)
            hidden = self.nbt.forward(hidden, turn_mask)

            hidden = self.attn2.forward(
                self.query_sent_layer_norm(hid_slot).view(-1, ts, self.bert_output_dim),
                hidden,
                hidden,
                mask=turn_mask)

            sigmoid = torch.sigmoid(self.gate(torch.cat([hidden_curr, hidden], -1)))
            hidden = sigmoid * hidden_curr + (1 - sigmoid) * hidden
            # passage_level merge
            if self.args.passage_level_combine:
                prev_label_hidden_passage = self.attn3.forward(hidden.view(-1, self.bert_output_dim), prev_label_hidden, prev_label_hidden)
                prev_label_hidden_passage = prev_label_hidden_passage.view(-1, ts, self.bert_output_dim)
                passage_history_gate = torch.sigmoid(self.passage_history_gate(torch.cat([hidden, prev_label_hidden_passage], -1)))
                hidden = (1 - passage_history_gate)*hidden + passage_history_gate*prev_label_hidden_passage

            loss_update = torch.Tensor([0.0]).to(self.device)
            acc_update = torch.Tensor([0.0]).to(self.device)
            if update is not None:
                # tanh activation
                hidden_update = torch.tanh(self.clsf_update(F.dropout(hidden, p=self.args.mt_drop, training=self.training)))
                prob_update = torch.sigmoid(self.update_linear(torch.cat([hidden_update, torch.cat([torch.zeros_like(hidden_update[:, :1]).cuda(), hidden_update[:, :-1]], 1)], -1))).squeeze()
                prob_update = prob_update.view(slot_dim, ds, ts).permute(1, 2, 0).contiguous()
                mask = (update > -1).float()
                update_1 = update.masked_fill((1 - mask).byte(), 1)
                prob_update_1 = prob_update.masked_fill((1 - mask).byte(), 1)
                loss_update = F.binary_cross_entropy(prob_update_1, update_1, reduction="none")
                loss_update = loss_update.sum() / mask.sum()
                tp = torch.sum((prob_update >= 0.5) & (update == 1))
                tn = torch.sum((prob_update < 0.5) & (update == 0))
                acc_update = (tp + tn) / mask.sum()

            hidden = self.sent_layer_norm(self.linear(self.dropout(hidden)))
            hidden = hidden.view(slot_dim, ds, ts, -1)

            # Label (slot-value) encoding
            main_loss = torch.Tensor([0.0]).to(self.device)
            loss_slot = []
            pred_slot = []
            output = []

            for s, slot_id in enumerate(target_slot): ## note: target_slot is successive
                # loss calculation
                hid_label = self.value_lookup[slot_id].weight
                num_slot_labels = hid_label.size(0)
                _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds*ts*num_slot_labels, -1)
                _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds*ts*num_slot_labels, -1)
                _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

                if self.distance_metric == "euclidean":
                    _dist = -_dist
                _, pred = torch.max(_dist, -1)
                pred_slot.append(pred.view(ds, ts, 1))
                output.append(_dist)

                if labels is not None:
                    _loss = self.nll(_dist.view(ds*ts, -1), labels[:, :, s].view(-1))
                    loss_slot.append(_loss.item())
                    main_loss += _loss

            if mt is True and update is not None:
                loss = self.lamb * main_loss + (1 - self.lamb) * loss_update
            else:
                loss = main_loss

            if labels is None:
                return output

            # calculate joint accuracy
            pred_slot = torch.cat(pred_slot, 2)
            accuracy = (pred_slot == labels).view(-1, slot_dim)
            acc_slot = torch.sum(accuracy, 0).float() \
                       / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
            acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
                  / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float() # joint accuracy

            if n_gpu == 1:
                return loss, loss_slot, acc, acc_slot, pred_slot, (main_loss, loss_update, acc_update)
            else:
                return loss.unsqueeze(0), loss.unsqueeze(0), acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot, main_loss.unsqueeze(0), loss_update.unsqueeze(0), acc_update.unsqueeze(0)

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)   # torch.nn.init.orthogonal_() ???
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)
