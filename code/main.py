import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import csv
import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
import json

import pdb

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from tensorboardX import SummaryWriter
from copy import deepcopy
import math

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, update=None, dial_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.update = update
        self.dial_id = dial_id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id, update, prev_label_id, dial_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id
        self.prev_label_id = prev_label_id
        self.update = update
        self.dial_id = dial_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':     # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        # MultiWOZ dataset
        if "data/multiwoz" in config.data_dir:
            fp_ontology = open(os.path.join(config.data_dir, "ontology.json"), "r")
            ontology = json.load(fp_ontology)
            for slot in ontology.keys():
                ontology[slot].append("none")
            fp_ontology.close()

            if not config.target_slot == 'all':
                slot_idx = {'attraction':'0:1:2', 'bus':'3:4:5:6', 'hospital':'7', 'hotel':'8:9:10:11:12:13:14:15:16:17',\
                            'restaurant':'18:19:20:21:22:23:24', 'taxi':'25:26:27:28', 'train':'29:30:31:32:33:34'}
                target_slot =[]
                for key, value in slot_idx.items():
                    if key != config.target_slot:
                        target_slot.append(value)
                config.target_slot = ':'.join(target_slot)

        else:
            raise NotImplementedError()

        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # lower case for ontology value
        for key in ontology.keys():
            ontology[key] = [x.lower() for x in ontology[key]]

        # select slots to train
        nslots = len(ontology.keys())
        self.nslots = nslots
        target_slot = list(ontology.keys())

        if config.target_slot == 'all':
            self.target_slot_idx = [*range(0, nslots)]
        else:
            self.target_slot_idx = sorted([int(x) for x in config.target_slot.split(':')])

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())
        for i, slot in enumerate(self.target_slot):
            slot_domain, slot_name = slot.split('-')
            if slot_name == "pricerange":
                self.target_slot[i] = "price range"
        logger.info('Processor: target_slot')
        logger.info(self.target_slot)

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", accumulation)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", accumulation)

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", accumulation)

    def get_labels(self):
        """See base class."""
        return [self.ontology[slot] for slot in self.target_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

            label = [line[4+idx] for idx in self.target_slot_idx]
            update = [line[4+self.nslots+idx] for idx in self.target_slot_idx]
            dial_id = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, update=update, dial_id=dial_id))
        return examples


class SUMBTDataset(Dataset):
    def __init__(self, examples, label_list, tokenizer, max_seq_length=64, max_turn_length=22):
        self.examples = examples
        self.label_list = label_list
        self.tokenizer = tokenizer
        label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]

        self.all_features = []
        prev_dialogue_idx = None

        max_turn = 0
        for (ex_index, example) in enumerate(examples):
            if max_turn < int(example.guid.split('-')[2]):
                max_turn = int(example.guid.split('-')[2])
        max_turn_length = min(max_turn+1, max_turn_length)
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
            tokens_b = None
            if example.text_b:
                tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            input_len = [len(tokens), 0]

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                input_len[1] = len(tokens_b) + 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            label_id = []
            label_info = 'label: '
            label_init = [label_map[i]['none'] for i in range(len(label_map))]

            for i, label in enumerate(example.label):
                if label == 'dontcare':
                    label = 'do not care'
                label_id.append(label_map[i][label])
                label_info += '%s (id = %d) ' % (label, label_map[i][label])

            curr_dialogue_idx = example.guid.split('-')[1]
            curr_turn_idx = int(example.guid.split('-')[2])
            if curr_turn_idx == 0:
                prev_label_id = label_init

            if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
                self.all_features.append(features)
                features = []

            if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_len=input_len,
                                  label_id=label_id,
                                  update=list(map(int, example.update)),
                                  prev_label_id=prev_label_id,
                                  dial_id=example.dial_id
                                  )
                )

            prev_dialogue_idx = curr_dialogue_idx
            prev_turn_idx = curr_turn_idx
            prev_label_id = label_id

        self.all_features.append(features)

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, index):
        input_ids = [f.input_ids for f in self.all_features[index]]
        max_len = max([len(i) for i in input_ids])
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [0] * (max_len-len(input_ids[i]))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_len = torch.tensor([f.input_len for f in self.all_features[index]], dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in self.all_features[index]], dtype=torch.long)
        label_prev_ids = torch.tensor([f.prev_label_id for f in self.all_features[index]], dtype=torch.long)
        update = torch.tensor([f.update for f in self.all_features[index]], dtype=torch.long)
        dial_ids = [f.dial_id for f in self.all_features[index]]
        return input_ids, input_len, label_ids, update, label_prev_ids, dial_ids


def collate_fn(batch):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        max_dim = max([i.size(1) for i in seq])
        result = torch.ones((len(seq), max_len, max_dim)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0), :seq[i].size(1)] = seq[i]
        return result

    input_ids_list, input_len_list, label_ids_list, update_list, label_prev_ids_list, dial_list = [], [], [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        input_len_list.append(i[1])
        label_ids_list.append(i[2])
        update_list.append(i[3])
        label_prev_ids_list.append(i[4])
        dial_list.append(i[5])

    input_ids = padding(input_ids_list, torch.LongTensor([0]))
    input_len = padding(input_len_list, torch.LongTensor([0]))
    label_ids = padding(label_ids_list, torch.LongTensor([-1]))
    update = padding(update_list, torch.LongTensor([-1])).float()
    label_prev_ids = padding(label_prev_ids_list, torch.LongTensor([-1]))
    return input_ids, input_len, label_ids, update, label_prev_ids, dial_list


def get_label_embedding(labels, max_seq_length, tokenizer, device, max_label_length_record):
    features = []
    for label in labels:
        label_tokens = ["[CLS]"] + tokenizer.tokenize(label) + ["[SEP]"]
        label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
        label_len = len(label_token_ids)
        if label_len > max_label_length_record:
            max_label_length_record = label_len
        if label_len > max_seq_length:
            label_token_ids = ["[CLS]"] + label_token_ids[1:max_seq_length-1] + ["[SEP]"]
            label_len = len(label_token_ids)
        label_padding = [0] * (max_seq_length - len(label_token_ids))
        label_token_ids += label_padding
        assert len(label_token_ids) == max_seq_length

        features.append((label_token_ids, label_len))

    all_label_token_ids = torch.tensor([f[0] for f in features], dtype=torch.long).to(device)
    all_label_len = torch.tensor([f[1] for f in features], dtype=torch.long).to(device)

    return all_label_token_ids, all_label_len, max_label_length_record


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_dir", default='/home/.pytorch_pretrained_bert',
                        type=str, required=False,
                        help="The directory of the pretrained BERT model")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train: bert-gru-sumbt, bert-lstm-sumbt"
                             "bert-label-embedding, bert-gru-label-embedding, bert-lstm-label-embedding")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--target_slot",
                        default='all',
                        type=str,
                        required=True,
                        help="Target slot idx to train model. ex. 'all', '0:1:2', or an excluding slot name 'attraction'" )
    parser.add_argument("--tf_dir",
                        default='tensorboard',
                        type=str,
                        required=False,
                        help="Tensorboard directory")
    parser.add_argument("--model",
                        default='BeliefTrackerSlotQueryMultiSlot',
                        type=str,
                        required=True,
                        help="model file name" )
    parser.add_argument("--mt_drop", type=float, default=0.)
    parser.add_argument("--fix_utterance_encoder",
                        action='store_true',
                        help="Do not train BERT utterance encoder")

    # Other parameters
    parser.add_argument("--window", default=1, type=int)
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_label_length",
                        default=20,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_turn_length",
                        default=22,
                        type=int,
                        help="The maximum total input turn length. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=100,
                        help="hidden dimension used in belief tracker")
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=1,
                        help="number of RNN layers")
    parser.add_argument('--zero_init_rnn',
                        action='store_true',
                        help="set initial hidden of rnns zero")
    parser.add_argument('--attn_head',
                        type=int,
                        default=4,
                        help="the number of heads in multi-headed attention")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_best_acc",
                        action='store_true')
    parser.add_argument("--combine",
                        type=str)
    parser.add_argument("--curr",
                        type=str,
                        default="attn")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--distance_metric",
                        type=str,
                        default="cosine",
                        help="The metric for distance between label embeddings: cosine, euclidean.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total dialog batch size for training.")
    parser.add_argument("--dev_batch_size",
                        default=1,
                        type=int,
                        help="Total dialog batch size for validation.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total dialog batch size for evaluation.")
    parser.add_argument("--lamb",
                        default=0.5,
                        type=float)
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for BertAdam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--patience",
                        default=10.0,
                        type=float,
                        help="The number of epochs to allow no further improvement.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--do_not_use_tensorboard",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--combine_sv_encode",
                        type=int,
                        default=1,
                        help="whether to use slot combine slot value as history"
                        )
    parser.add_argument("--utterance_level_combine",
                        type=int,
                        default=0,
                        help="whether to use last dialogue state in token level"
                        )
    parser.add_argument("--passage_level_combine",
                        type=int,
                        default=0,
                        help="whether to use last dialogue state in utterance level"
                        )
    parser.add_argument("--mix_teaching_force",
                        type=int,
                        default=0,
                        help="mix teaching force or not"
                        )
    parser.add_argument("--schedule_sampling",
                        type=int,
                        default=0,
                        help="mix teaching force or not"
                        )
    parser.add_argument("--sample_k",
                        type=float,
                        default=1.0,
                        help="mix teaching force or not"
                        )
    parser.add_argument("--complete_no_teaching_force",
                        type=int,
                        default=0,
                        help="complete_no_teaching_force"
                        )

    args = parser.parse_args()
    sample_k = args.sample_k
    if args.mix_teaching_force:
        args.mix_teaching_force = True
        random_bench = 0.5
    else:
        args.mix_teaching_force = False
        random_bench = 1

    print('Mix Teaching Force: {}'.format(args.mix_teaching_force))
    print('Passage Level Combine: {}'.format(args.passage_level_combine))
    print('Utterance Level Combine: {}'.format(args.utterance_level_combine))

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.do_train and not args.do_eval and not args.do_analyze:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # Tensorboard logging
    if not args.do_not_use_tensorboard:
        tb_file_name = args.output_dir.split('/')[1]
        summary_writer = SummaryWriter("./%s/%s" % (args.tf_dir, tb_file_name))
    else:
        summary_writer = None

    # Logger
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, "%s.txt"%(tb_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # CUDA setup
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    # Get Processor
    processor = Processor(args)

    label_list = processor.get_labels()
    num_labels = [len(labels) for labels in label_list] # number of slot-values in each slot-type

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, accumulation=accumulation)
        dev_examples = processor.get_dev_examples(args.data_dir, accumulation=accumulation)

        # Training utterances
        train_dataset = SUMBTDataset(train_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

        num_train_batches = len(train_dataset)
        num_train_steps = int(num_train_batches / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True, num_workers=16, collate_fn=lambda x: collate_fn(x))

        # Dev utterances
        dev_dataset = SUMBTDataset(dev_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

        logger.info("***** Running validation *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.dev_batch_size)

        dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size, drop_last=False, num_workers=8, collate_fn=lambda x: collate_fn(x))

    logger.info("Loaded data!")

    # Build the models
    # Prepare model
    BeliefTracker = getattr(__import__(args.model), 'BeliefTracker')
    model = BeliefTracker(args, num_labels, device)

    if args.fp16:
        model.half()

    # reload_model
    try:
        output_model_file = os.path.join(args.output_dir, "step_model")
        ptr_model = torch.load(output_model_file)
        for key in deepcopy(ptr_model):
            if 'slot_lookup' in key or 'value_lookup' in key or 'sv_encoder' in key:
                ptr_model.pop(key)
        model.load_state_dict(ptr_model, strict=False)
        print("#######reload_from\t{}#########".format(output_model_file))
    except:
        pass
    model.to(device)
    save_configure(args, num_labels, processor.ontology)

    # Get slot-value embeddings
    label_token_ids, label_len = [], []
    max_label_length_record = 0
    for labels in label_list:
        token_ids, lens, max_label_length_record = get_label_embedding(labels, args.max_label_length, tokenizer, device, max_label_length_record)
        label_token_ids.append(token_ids)
        label_len.append(lens)

    # Get domain-slot-type embeddings
    slot_token_ids, slot_len, max_label_length_record = get_label_embedding(processor.target_slot, args.max_label_length, tokenizer, device, max_label_length_record)
    # Initialize slot and value embeddings
    model.initialize_slot_value_lookup(label_token_ids, slot_token_ids, args.combine_sv_encode)

    # Data parallelize when use multi-gpus
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.learning_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.learning_rate},
            ]
            return optimizer_grouped_parameters
        if n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(model.module)

        t_total = num_train_steps

        if args.local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)
        logger.info(optimizer)

    ###############################################################################
    # Training code
    ###############################################################################

    if args.do_train:
        logger.info("Training...")

        global_step = 0
        last_update = 0
        best_loss = None
        best_acc = 0
        train_mt = True

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            # Train
            model.train()
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                # eval evey 500 steps
                if (step) % 500 == 0:
                    model.eval()
                    dev_acc_ = 0
                    dev_loss_ = 0
                    nb_dev_examples, nb_dev_steps = 0, 0
                    #################################
                    for _, batch_ in enumerate(tqdm(dev_dataloader)):
                        batch_ = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch_)
                        input_ids, input_len, label_ids, update, prev_label_ids, dial_ids = batch_
                        if input_ids.dim() == 2:
                            input_ids = input_ids.unsqueeze(0)
                            input_len = input_len.unsqueeze(0)
                            label_ids = label_ids.unsuqeeze(0)
                            update = update.unsqueeze(0)
                            prev_label_ids = prev_label_ids.unsuqeeze(0)
                        with torch.no_grad():
                            loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids,
                                                                                   input_len,
                                                                                   label_ids,
                                                                                   update,
                                                                                   n_gpu,
                                                                                   prev_label_ids=prev_label_ids,
                                                                                   no_teaching_force=args.mix_teaching_force,
                                                                                   combine_sv_encode=args.combine_sv_encode)
                        num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                        dev_loss_ += loss.item() * num_valid_turn
                        dev_acc_ += acc.item() * num_valid_turn
                        nb_dev_examples += num_valid_turn

                    dev_acc_ = dev_acc_ / nb_dev_examples
                    dev_loss_ = dev_loss_/nb_dev_examples

                    model.train()
                    if best_loss is None or dev_loss_ < best_loss:
                        best_loss = dev_loss_
                    if dev_acc_ > best_acc:
                        print('#' * 20+'save_step_model  :{}'.format(best_acc)+'#' * 20)
                        output_model_file = os.path.join(args.output_dir, "step_model")
                        torch.save(model.state_dict(), output_model_file)
                        best_acc = dev_acc_
                        last_update = epoch
                    acc_loss_file = open(os.path.join(args.output_dir, "acc_loss.txt"), "a", encoding="utf-8")
                    acc_loss_file.write(
                        'epoch: {}  step:{}  acc:{}  loss:{}  best_acc:{}  best_loss: {}  epoch_end: {}\n'.format(epoch, step,
                                                                                                    dev_acc_,
                                                                                                    dev_loss_,
                                                                                                    best_acc,
                                                                                                    best_loss, 'False'))
                    acc_loss_file.close()

                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, update, prev_label_ids, dial_ids = batch

                # Forward
                if args.schedule_sampling and args.mix_teaching_force:
                    random_bench = sample_k / float(sample_k + math.exp(epoch/sample_k))

                if args.complete_no_teaching_force:
                    random_bench = 0.0

                no_teaching_force = random.random() > random_bench
                if no_teaching_force:
                    model.eval()
                    with torch.no_grad():
                        _, _, _, _, pred_slot, _ = model(input_ids, input_len,
                                                                    label_ids,
                                                                    update,
                                                                    n_gpu,
                                                                    mt=train_mt,
                                                                    prev_label_ids=prev_label_ids,
                                                                    no_teaching_force=True,
                                                                    combine_sv_encode=args.combine_sv_encode)
                        prev_label_ids[:, 1:, :] = pred_slot[:, 0:-1, :]
                    model.train()
                loss, loss_slot, acc, acc_slot, _, tup = model(input_ids,
                                                                input_len,
                                                                label_ids,
                                                                update,
                                                                n_gpu,
                                                                mt=train_mt,
                                                                prev_label_ids=prev_label_ids,
                                                                no_teaching_force=False,
                                                                combine_sv_encode=args.combine_sv_encode)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Backward
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                # tensrboard logging
                if summary_writer is not None:
                    summary_writer.add_scalar("Epoch", epoch, global_step)
                    summary_writer.add_scalar("Train/Loss", loss, global_step)
                    summary_writer.add_scalar("Train/JointAcc", acc, global_step)
                    summary_writer.add_scalar("Train/main_loss", tup[0], global_step)
                    summary_writer.add_scalar("Train/update_loss", tup[1], global_step)
                    summary_writer.add_scalar("Train/update_acc", tup[2], global_step)
                    if n_gpu == 1:
                        for i, slot in enumerate(processor.target_slot):
                            summary_writer.add_scalar("Train/Loss_%s" % slot.replace(' ', '_'), loss_slot[i], global_step)
                            summary_writer.add_scalar("Train/Acc_%s" % slot.replace(' ', '_'), acc_slot[i], global_step)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Perform evaluation on validation dataset
            badcase_list = []
            model.eval()
            dev_loss = 0
            dev_acc = 0
            dev_loss_slot, dev_acc_slot = None, None
            dev_tup = [0, 0, 0]
            nb_dev_examples, nb_dev_steps = 0, 0

            for _, batch in enumerate(tqdm(dev_dataloader)):
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids, input_len, label_ids, update, prev_label_ids, dial_ids = batch
                if input_ids.dim() == 2:
                    input_ids = input_ids.unsqueeze(0)
                    input_len = input_len.unsqueeze(0)
                    label_ids = label_ids.unsuqeeze(0)
                    update = update.unsqueeze(0)
                    prev_label_ids = prev_label_ids.unsqueeze(0)

                with torch.no_grad():
                    loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids,
                                                                           input_len,
                                                                           label_ids,
                                                                           update,
                                                                           n_gpu,
                                                                           prev_label_ids=prev_label_ids,
                                                                           no_teaching_force=args.mix_teaching_force,
                                                                           combine_sv_encode=args.combine_sv_encode)

                badcase = (pred_slot != label_ids) * (label_ids > -1)
                badcase_1 = badcase.sum(-1).nonzero()
                for b in badcase_1:
                    b_idx = b.cpu().numpy().tolist()
                    sent = ' '.join(tokenizer.convert_ids_to_tokens(input_ids[b_idx[0], b_idx[1]].cpu().numpy().tolist()))
                    pred = ' '.join([label_list[i][j] for i, j in enumerate(pred_slot[b_idx[0], b_idx[1]].cpu().numpy().tolist())])
                    gold = ' '.join([label_list[i][j] for i, j in enumerate(label_ids[b_idx[0], b_idx[1]].cpu().numpy().tolist())])
                    badcase_list.append(f"{sent}\t{pred}\t{gold}\n")
                num_valid_turn = torch.sum(label_ids[:, :, 0].view(-1) > -1, 0).item()
                dev_loss += loss.item() * num_valid_turn
                dev_acc += acc.item() * num_valid_turn
                dev_tup[0] += tup[0] * num_valid_turn
                dev_tup[1] += tup[1] * num_valid_turn
                dev_tup[2] += tup[2] * num_valid_turn

                if n_gpu == 1:
                    if dev_loss_slot is None:
                        dev_loss_slot = [l * num_valid_turn for l in loss_slot]
                        dev_acc_slot = acc_slot * num_valid_turn
                    else:
                        for i, l in enumerate(loss_slot):
                            dev_loss_slot[i] = dev_loss_slot[i] + l * num_valid_turn
                        dev_acc_slot += acc_slot * num_valid_turn

                nb_dev_examples += num_valid_turn

            dev_loss = dev_loss / nb_dev_examples
            dev_acc = dev_acc / nb_dev_examples
            dev_tup = list(map(lambda x: x / nb_dev_examples, dev_tup))
            #train_mt = dev_tup[2] < 0.95

            if n_gpu == 1:
                dev_acc_slot = dev_acc_slot / nb_dev_examples

            # tensorboard logging
            if summary_writer is not None:
                summary_writer.add_scalar("Validate/Loss", dev_loss, global_step)
                summary_writer.add_scalar("Validate/Acc", dev_acc, global_step)
                summary_writer.add_scalar("Validate/main_loss", dev_tup[0], global_step)
                summary_writer.add_scalar("Validate/update_loss", dev_tup[1], global_step)
                summary_writer.add_scalar("Validate/update_acc", dev_tup[2], global_step)
                if n_gpu == 1:
                    for i, slot in enumerate(processor.target_slot):
                        summary_writer.add_scalar("Validate/Loss_%s" % slot.replace(' ', '_'), dev_loss_slot[i]/nb_dev_examples, global_step)
                        summary_writer.add_scalar("Validate/Acc_%s" % slot.replace(' ', '_'), dev_acc_slot[i], global_step)

            dev_loss = round(dev_loss, 6)

            if dev_acc > best_acc:
                badcase_fp = open(os.path.join(args.output_dir, "dev_badcase.txt"), "w", encoding="utf-8")
                for line in badcase_list:
                    badcase_fp.write(line)
                badcase_fp.close()
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, "step_model")
                if args.do_train:
                    if n_gpu == 1:
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        torch.save(model.module.state_dict(), output_model_file)
                best_acc = dev_acc
                last_update = epoch

            if best_loss is None or dev_loss < best_loss:
                best_loss = dev_loss

            acc_loss_file = open(os.path.join(args.output_dir, "acc_loss.txt"), "a", encoding="utf-8")
            acc_loss_file.write(
                'epoch: {}  step:{}  acc:{}  loss:{}  best_acc:{}  best_loss:{}  epoch_end: {}\n'.format(epoch, step, dev_acc, dev_loss, best_acc, best_loss, 'True'))
            acc_loss_file.close()

            if last_update + args.patience <= epoch:
                break

    # Evaluation
    # Load a trained model that you have fine-tuned

    output_model_file = os.path.join(args.output_dir, "step_model")
    model = BeliefTracker(args, num_labels, device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # in the case that slot and values are different between the training and evaluation
    ptr_model = torch.load(output_model_file)

    if n_gpu == 1:
        model.load_state_dict(ptr_model, strict=False)
    else:
        print("Evaluate using only one device!")
        model.module.load_state_dict(ptr_model)

    model.to(device)
    # Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        eval_examples = processor.get_test_examples(args.data_dir, accumulation=accumulation)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataset = SUMBTDataset(eval_examples, label_list, tokenizer, max_seq_length=args.max_seq_length, max_turn_length=args.max_turn_length)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=lambda x: collate_fn(x))

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        eval_update_acc = 0
        eval_loss_slot, eval_acc_slot = None, None
        nb_eval_steps, nb_eval_examples = 0, 0

        accuracies = {'joint7': 0, 'slot7': 0, 'joint5': 0, 'slot5': 0, 'joint_rest': 0, 'slot_rest': 0,
                      'num_turn': 0, 'num_slot7': 0, 'num_slot5': 0, 'num_slot_rest': 0}

        dial_bad_info = {}
        tmp_dialog = {}
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, input_len, label_ids, update, prev_label_ids, dial_ids = batch
            if input_ids.dim() == 2:
                input_ids = input_ids.unsqueeze(0)
                input_len = input_len.unsqueezen(0)
                label_ids = label_ids.unsuqeeze(0)
                update = update.unsqueeze(0)
                prev_label_ids = prev_label_ids.unsqueeze(0)

            with torch.no_grad():
                loss, loss_slot, acc, acc_slot, pred_slot, tup = model(input_ids,
                                                                       input_len,
                                                                       label_ids,
                                                                       update,
                                                                       n_gpu,
                                                                       prev_label_ids=prev_label_ids,
                                                                       no_teaching_force=args.mix_teaching_force,
                                                                       combine_sv_encode=args.combine_sv_encode)

            badcase = (pred_slot != label_ids) * (label_ids > -1)
            badcase_1 = badcase.sum(-1).nonzero()
            dial_len = [len(x) for x in dial_ids]
            dial_index_bad_list = []
            tmp_dial_bad_info = {}
            dial_bad_id = ''
            if True:
                dialog_id = dial_ids[0][0]
                tmp_dialog[dialog_id] = []
                for dial_turn_id, dial_turn in enumerate(input_ids.tolist()[0]):
                    tmp = {}
                    seq_len_list = input_len[0][dial_turn_id].tolist()
                    seq_len = seq_len_list[0] + seq_len_list[1]
                    tmp['sent'] = ' '.join(tokenizer.convert_ids_to_tokens(input_ids[0, dial_turn_id][0:seq_len].cpu().numpy().tolist()))
                    tmp['turn_id'] = dial_turn_id
                    tmp['pred'] = [label_list[i][j] for i, j in enumerate(pred_slot[0, dial_turn_id].cpu().numpy().tolist())]
                    tmp['label'] = [label_list[i][j] for i, j in enumerate(label_ids[0, dial_turn_id].cpu().numpy().tolist())]
                    tmp_dialog[dialog_id].append(tmp)

            for b in badcase_1:
                b_idx = b.cpu().numpy().tolist()
                if b_idx[0] not in dial_index_bad_list:
                    dial_index_bad_list.append(b_idx[0])
                    dial_bad_id = dial_ids[b_idx[0]][0]
                    tmp_dial_bad_info[dial_bad_id] = {}
                    tmp_dial_bad_info[dial_bad_id]['bad_turn'] = [b_idx[1]]
                    tmp_dial_bad_info[dial_bad_id]['dial_len'] = dial_len[b_idx[0]]
                    tmp_dial_bad_info[dial_bad_id]['history'] = []
                    tmp_dial_bad_info[dial_bad_id]['tmp_id'] = b_idx[0]
                else:
                    tmp_dial_bad_info[dial_bad_id]['bad_turn'].append(b_idx[1])
            for key in tmp_dial_bad_info.keys():
                tmp_id = tmp_dial_bad_info[key]['tmp_id']
                bad_turn = tmp_dial_bad_info[key]['bad_turn']
                dial_len = tmp_dial_bad_info[key]['dial_len']
                for turn_id in range(dial_len):
                    tmp_turn_info = {
                        'sent': ' '.join(
                            tokenizer.convert_ids_to_tokens(input_ids[tmp_id, turn_id].cpu().numpy().tolist())),
                        'pred': '\t'.join(
                            [label_list[i][j] for i, j in enumerate(pred_slot[tmp_id, turn_id].cpu().numpy().tolist())])
                    }
                    if turn_id in bad_turn:
                        tmp_turn_info['gold'] = '\t'.join(
                            [label_list[i][j] for i, j in enumerate(label_ids[tmp_id, turn_id].cpu().numpy().tolist())])
                    tmp_dial_bad_info[key]['history'].append(tmp_turn_info)
                tmp_dial_bad_info[key].pop('tmp_id')
            dial_bad_info.update(tmp_dial_bad_info)

            accuracies = eval_all_accs(pred_slot, label_ids, accuracies)

            nb_eval_ex = (label_ids[:, :, 0].view(-1) != -1).sum().item()
            nb_eval_examples += nb_eval_ex
            nb_eval_steps += 1
            eval_update_acc += tup[2] * nb_eval_ex

            if n_gpu == 1:
                eval_loss += loss.item() * nb_eval_ex
                eval_accuracy += acc.item() * nb_eval_ex
                if eval_loss_slot is None:
                    eval_loss_slot = [l * nb_eval_ex for l in loss_slot]
                    eval_acc_slot = acc_slot * nb_eval_ex
                else:
                    for i, l in enumerate(loss_slot):
                        eval_loss_slot[i] = eval_loss_slot[i] + l * nb_eval_ex
                    eval_acc_slot += acc_slot * nb_eval_ex
            else:
                eval_loss += sum(loss) * nb_eval_ex
                eval_accuracy += sum(acc) * nb_eval_ex

        badcase_out_file_name = 'eval_bad_case'
        badcase_out_file = os.path.join(args.output_dir, "%s.json" % badcase_out_file_name)
        with open(badcase_out_file, 'w') as f:
            json.dump(dial_bad_info, f, indent=4)
        # out_file = os.path.join(args.output_dir, "%s.json" % 'out_result_taxi')
        # with open(out_file, 'w') as f:
        #     json.dump(tmp_dialog, f, indent=4)

        eval_update_acc = eval_update_acc / nb_eval_examples
        eval_loss = eval_loss / nb_eval_examples
        eval_accuracy = eval_accuracy / nb_eval_examples
        if n_gpu == 1:
            eval_acc_slot = eval_acc_slot / nb_eval_examples

        loss = tr_loss / nb_tr_steps if args.do_train else None

        if n_gpu == 1:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss,
                      'eval_loss_slot':'\t'.join([ str(val/ nb_eval_examples) for val in eval_loss_slot]),
                      'eval_acc_slot':'\t'.join([ str((val).item()) for val in eval_acc_slot])
                        }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'loss': loss
                      }

        out_file_name = 'eval_results'
        if args.target_slot == 'all':
            out_file_name += '_all'
        output_eval_file = os.path.join(args.output_dir, "%s.txt" % out_file_name)

        if n_gpu == 1:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        out_file_name = 'eval_all_accuracies'
        with open(os.path.join(args.output_dir, "%s.txt" % out_file_name), 'w') as f:
            f.write('joint acc (7 domain) : slot acc (7 domain) : joint acc (5 domain): slot acc (5 domain): joint restaurant : slot acc restaurant \n')
            f.write('%.5f : %.5f : %.5f : %.5f : %.5f : %.5f \n' % (
                (accuracies['joint7']/accuracies['num_turn']).item(),
                (accuracies['slot7']/accuracies['num_slot7']).item(),
                (accuracies['joint5']/accuracies['num_turn']).item(),
                (accuracies['slot5'] / accuracies['num_slot5']).item(),
                (accuracies['joint_rest']/accuracies['num_turn']).item(),
                (accuracies['slot_rest'] / accuracies['num_slot_rest']).item()
            ))


def eval_all_accs(pred_slot, labels, accuracies):

    def _eval_acc(_pred_slot, _labels):
        slot_dim = _labels.size(-1)
        accuracy = (_pred_slot == _labels).view(-1, slot_dim)
        num_turn = torch.sum(_labels[:, :, 0].view(-1) > -1, 0).float()
        num_data = torch.sum(_labels > -1).float()
        # joint accuracy
        joint_acc = sum(torch.sum(accuracy, 1) / slot_dim).float()
        # slot accuracy
        slot_acc = torch.sum(accuracy).float()
        return joint_acc, slot_acc, num_turn, num_data

    # 7 domains
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot, labels)
    accuracies['joint7'] += joint_acc
    accuracies['slot7'] += slot_acc
    accuracies['num_turn'] += num_turn
    accuracies['num_slot7'] += num_data

    # restaurant domain
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot[:,:,18:25], labels[:,:,18:25])
    accuracies['joint_rest'] += joint_acc
    accuracies['slot_rest'] += slot_acc
    accuracies['num_slot_rest'] += num_data

    pred_slot5 = torch.cat((pred_slot[:,:,0:3], pred_slot[:,:,8:]), 2)
    label_slot5 = torch.cat((labels[:,:,0:3], labels[:,:,8:]), 2)

    # 5 domains (excluding bus and hotel domain)
    joint_acc, slot_acc, num_turn, num_data = _eval_acc(pred_slot5, label_slot5)
    accuracies['joint5'] += joint_acc
    accuracies['slot5'] += slot_acc
    accuracies['num_slot5'] += num_data

    return accuracies


def save_configure(args, num_labels, ontology):
    with open(os.path.join(args.output_dir, "config.json"),'w') as outfile:
        data = { "hidden_dim": args.hidden_dim,
                "num_rnn_layers": args.num_rnn_layers,
                "zero_init_rnn": args.zero_init_rnn,
                "max_seq_length": args.max_seq_length,
                "max_label_length": args.max_label_length,
                "num_labels": num_labels,
                "attn_head": args.attn_head,
                 "distance_metric": args.distance_metric,
                 "fix_utterance_encoder": args.fix_utterance_encoder,
                 "task_name": args.task_name,
                 "bert_dir": args.bert_dir,
                 "bert_model": args.bert_model,
                 "do_lower_case": args.do_lower_case,
                 "ontology": ontology}
        json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    main()
