'''TODO
- Incorporate all cfg to get_loss function
- Create a new MergeDatasetReader to yield both sup and unsup data || Done
- Write a new Trainer to get both sup data and up sup data || Not pursued
- Incorporate get_loss to BaselineModel
    - Incorporate sup batch loss || Done
    - Incorporate unsup batch loss || Done
- Create unsup batch || Done
- Change cfg in get_loss function to get the function running || Done
- unsup_criterion = nn.KLDivLoss(reduction='none') || Done
'''

# %%
'''Dependencies for UDA'''
import copy

import fire

import torch
import torch.nn as nn
import torch.nn.functional as F

# import models
# import train
# from load_data import load_data
# from utils.utils import set_seeds, get_device, _get_device, torch_device_one
# from utils import optim, configuration

# %%
'''Utils'''
def torch_device_one():
    return torch.tensor(1.).to(_get_device())

def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# %%
'''Dependencies for AllenNLP model'''
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '/Users/kchu/Documents/Projects/Senior Project/Claim Extraction/detecting-scientific-claim-master/')

from typing import Iterator, List, Dict, Optional
import os
import json
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from allennlp.common.util import JsonDict

import torch
import torch.optim as optim
from torch.nn import ModuleList
import torch.nn.functional as F

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from allennlp.data.fields import Field, TextField, LabelField, ListField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
import torch.nn as nn

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler

from allennlp.modules import Seq2VecEncoder, TimeDistributed, TextFieldEmbedder, ConditionalRandomField, FeedForward
from torch.nn.modules.linear import Linear
import allennlp.nn.util as util

EMBEDDING_DIM = 300
TRAIN_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/train_labels.json'
# TRAIN_PATH = './train_augmented_labels.json'
COMBINED_TRAIN_PATH = './combined_sup_train_data_unsup_dev_pubmed_data.json'
VALIDATION_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/validation_labels.json'
TEST_PATH = 'https://s3-us-west-2.amazonaws.com/pubmed-rct/test_labels.json'
# DISCOURSE_MODEL_PATH = './output_crf_pubmed_rct_glove/model.tar.gz'
# archive = load_archive(DISCOURSE_MODEL_PATH)
# discourse_predictor = Predictor.from_archive(archive, 'discourse_crf_predictor')

# %%
'''Temporary configuration'''
from argparse import Namespace

cfg = Namespace(
    tsa = None,
    uda_coeff = 1,
    uda_softmax_temp = 1.,
    uda_confidence_thresh = -1,
)

# %%
# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())

# %%
unsup_criterion = nn.KLDivLoss(reduction='none')
sup_criterion = nn.CrossEntropyLoss(reduction='none')

# %%
'''Main'''
class ClaimAnnotationReaderJSON(DatasetReader):
    """
    Reading annotation dataset in the following JSON format:

    {
        "paper_id": ..., 
        "user_id": ...,
        "sentences": [..., ..., ...],
        "labels": [..., ..., ...] 
    }
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    # @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']
                yield self.text_to_instance(sents, labels)

    # @overrides
    def text_to_instance(self,
                         sents: List[str],
                         labels: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        
        if labels is not None:
            fields['labels'] = SequenceLabelField(labels, sentence_sequence)
        return Instance(fields)

# %%
'''Create a MergeDatasetReader'''
class MergeDatasetReader(DatasetReader):
    """
    Reading annotation dataset in the following JSON format:

    {
        "paper_id": ..., 
        "user_id": ...,
        "sentences": [..., ..., ...],
        "labels": [..., ..., ...] 
    }
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    # @overrides
    def _read(self, sup_file_path, ori_unsup_file_path = None, aug_unsup_file_path = None):
        sup_file_path = cached_path(sup_file_path)
        # ori_unsup_file_path = cached_path(ori_unsup_file_path)
        # aug_unsup_file_path = cached_path(aug_unsup_file_path)

        sup_file_data = []
        sup_file_labels = []
        ori_unsup_file_data = []
        aug_unsup_file_data = []

        with open(sup_file_path, 'r') as file:
            for line in file:
                example = json.loads(line)
                sents = example['sentences']
                labels = example['labels']
                ori_unsup_sents = example['ori_unsup_sentences']
                aug_unsup_sents = example['aug_unsup_sentences']
                
                sup_file_data.append(sents)
                sup_file_labels.append(labels)
                ori_unsup_file_data.append(ori_unsup_file_data)
                aug_unsup_file_data.append(aug_unsup_sents)

                yield self.text_to_instance(sents, labels, ori_unsup_sents, aug_unsup_sents)
        
        # with open(sup_file_path, 'r') as file:
        #     for line in file:
        #         example = json.loads(line)
        #         sents = example['sentences']
        #         labels = example['labels']
        #         sup_file_data.append(sents)
        #         sup_file_labels.append(labels)
                
        # with open(sup_file_path, 'r') as file:
        #     for line in file:
        #         example = json.loads(line)
        #         sents = example['sentences']
        #         labels = example['labels']
        #         sup_file_data.append(sents)
        #         sup_file_labels.append(labels)

        # for i, data in enumerate(sup_file_data):        
        #     yield self.text_to_instance(sup_file_data[i], sup_file_labels[i], sup_file_data[i], sup_file_data[i])

    # @overrides
    def text_to_instance(self,
                         sup_sents: List[str],
                         sup_labels: List[str] = None,
                         ori_unsup_sents: List[str] = None,
                         aug_unsup_sents: List[str] = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_sents = [self._tokenizer.tokenize(sent) for sent in sup_sents]
        sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in tokenized_sents])
        fields['sentences'] = sentence_sequence
        
        if sup_labels is not None:
            fields['labels'] = SequenceLabelField(sup_labels, sentence_sequence)
        
        if ori_unsup_sents is not None and aug_unsup_sents is not None:
            # Create TextField for ori_unsup_sentences
            ori_unsup_tokenized_sents = [self._tokenizer.tokenize(sent) for sent in ori_unsup_sents]
            ori_unsup_sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in ori_unsup_tokenized_sents])

            # Create TextField for aug_unsup_sentences
            aug_unsup_tokenized_sents = [self._tokenizer.tokenize(sent) for sent in aug_unsup_sents]
            aug_unsup_sentence_sequence = ListField([TextField(tk, self._token_indexers) for tk in aug_unsup_tokenized_sents])

            fields['ori_unsup_sentences'] = ori_unsup_sentence_sequence
            fields['aug_unsup_sentences'] = aug_unsup_sentence_sequence
        # Fake data
        # fields['ori_unsup_sentences'] = sentence_sequence
        # fields['aug_unsup_sentences'] = sentence_sequence

        return Instance(fields)

# %%
token_indexer = PretrainedBertIndexer(
    pretrained_model="./biobert_v1.1_pubmed/vocab.txt",
    do_lowercase=True,
)

# %%
reader = ClaimAnnotationReaderJSON(
    token_indexers={"tokens": token_indexer},
    lazy=True
)
merge_reader = MergeDatasetReader(
    token_indexers={"tokens": token_indexer},
    lazy=True
)

# train_dataset = reader.read(TRAIN_PATH)
train_dataset = merge_reader.read(COMBINED_TRAIN_PATH)
validation_dataset = reader.read(VALIDATION_PATH)
test_dataset = reader.read(TEST_PATH)
# %%
vocab = Vocabulary()

vocab._token_to_index['labels'] = {'0': 0, '1': 1}

# %%
"""Prepare iterator"""
from allennlp.data.iterators import BasicIterator

iterator = BasicIterator(batch_size=8)

iterator.index_with(vocab)

# %%
# Loss function
def multiple_target_CrossEntropyLoss(logits, labels):
    loss = 0
    for i in range(logits.shape[0]):
        loss = loss + nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0]).cuda())(logits[i, :, :], labels[i, :])
    return loss / logits.shape[0]

# %%
"""Prepare the model"""
class BaselineModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BaselineModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def get_loss(self, sup_batch, sup_batch_labels, ori_unsup_input = None, aug_unsup_input = None, global_step = 0):
        # logits -> prob(softmax) -> log_prob(log_softmax)

        # batch
        # input_ids, segment_ids, input_mask, label_ids = sup_batch
        # input_ids = sup_batch['tokens']
        # segment_ids = sup_batch['tokens-type-ids']
        # input_mask = sup_batch['mask']
        sentences = sup_batch
        label_ids = sup_batch_labels
        # print('Sentence shape:', sentences['tokens'].shape)

        # if unsup_batch:
        #     # ori_input_ids, ori_segment_ids, ori_input_mask, \
        #     # aug_input_ids, aug_segment_ids, aug_input_mask  = unsup_batch
        #     ori_unsup_input, aug_unsup_input = unsup_batch

        #     # input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
        #     # segment_ids = torch.cat((segment_ids, aug_segment_ids), dim=0)
        #     # input_mask = torch.cat((input_mask, aug_input_mask), dim=0)
        #     # sentences['tokens'] = torch.cat((sup_batch['tokens'], aug_unsup_input['tokens']), dim=0)
        #     # sentences['tokens-offsets'] = torch.cat((sup_batch['tokens-offsets'], aug_unsup_input['tokens-offsets']), dim=0)
        #     # sentences['tokens-type-ids'] = torch.cat((sup_batch['tokens-type-ids'], aug_unsup_input['tokens-type-ids']), dim=0)
        #     # sentences['mask'] = torch.cat((sup_batch['mask'], aug_unsup_input['mask']), dim=0)
        #     # print('New sentence shape:', sentences['tokens'].shape)

        # logits
        # logits = model(input_ids, segment_ids, input_mask)
        embedded_sentence = self.text_field_embedder(sentences)
        # print('Embedded size:', embedded_sentence.size()) # (batch_size, num_sentences, seq_len, embedding_size)
        sentence_mask = util.get_text_field_mask(sentences)
        # print('Sentence mask:', sentence_mask.size()) # (batch_size, num_sentences, seq_len)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)
        # print('Encoded sentence:', encoded_sentence.size()) # (batch_size, num_sentences, embedding_size)

        logits = self.classifier_feedforward(encoded_sentence) # (batch_size, num_sentences, num_labels(2))
        logits = logits.squeeze(-1) # Actually doesnt do anything (batch_size, num_sentences, num_labels)
        
        # sup loss
        sup_size = label_ids.shape[0]            
        # sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        sup_loss = multiple_target_CrossEntropyLoss(logits[:sup_size], label_ids)

        if cfg.tsa: # None for now
            tsa_thresh = get_tsa_thresh(cfg.tsa, global_step, cfg.total_steps, start=1./logits.shape[-1], end=1)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one())
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if aug_unsup_input:
            unsup_size = aug_unsup_input['tokens'].shape[0]
            # ori
            with torch.no_grad():
                # Calculate logits for augmented unsup batch
                aug_embedded_sentence = self.text_field_embedder(aug_unsup_input) # (batch_size, num_sentences, seq_len, embedding_size)
                aug_sentence_mask = util.get_text_field_mask(aug_unsup_input) # (batch_size, num_sentences, seq_len)
                aug_encoded_sentence = self.sentence_encoder(aug_embedded_sentence, aug_sentence_mask) # (batch_size, num_sentences, embedding_size)

                aug_logits = self.classifier_feedforward(aug_encoded_sentence) # (batch_size, num_sentences, num_labels(2))
                aug_logits = aug_logits.squeeze(-1) # Actually doesnt do anything (batch_size, num_sentences, num_labels)
                aug_prob   = F.log_softmax(aug_logits, dim=-1)    # KLdiv target

                # Calculate logits for original unsup batch
                # ori_logits = model(ori_input_ids, ori_segment_ids, ori_input_mask)
                ori_embedded_sentence = self.text_field_embedder(ori_unsup_input) # (batch_size, num_sentences, seq_len, embedding_size)
                ori_sentence_mask = util.get_text_field_mask(ori_unsup_input) # (batch_size, num_sentences, seq_len)
                ori_encoded_sentence = self.sentence_encoder(ori_embedded_sentence, ori_sentence_mask) # (batch_size, num_sentences, embedding_size)

                ori_logits = self.classifier_feedforward(ori_encoded_sentence) # (batch_size, num_sentences, num_labels(2))
                ori_logits = ori_logits.squeeze(-1) # Actually doesnt do anything (batch_size, num_sentences, num_labels)
                ori_prob   = F.softmax(ori_logits, dim=-1)    # KLdiv target
                # ori_log_prob = F.log_softmax(ori_logits, dim=-1)
                
                # confidence-based masking
                if cfg.uda_confidence_thresh != -1: # == -1 for now
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg.uda_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones((unsup_size, 1), dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())

            # aug
            # softmax temperature controlling
            
            uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.

            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html
                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch_device_one())
            unsup_loss = torch.mean(unsup_loss)

            final_loss = sup_loss + cfg.uda_coeff * unsup_loss

            return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    def forward(self,
                sentences: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                ori_unsup_sentences: Dict[str, torch.LongTensor] = None,
                aug_unsup_sentences: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # print('Sentences:', sentences['tokens'].size()) # (batch_size, num_sentences, seq_len)
        embedded_sentence = self.text_field_embedder(sentences)
        # print('Embedded size:', embedded_sentence.size()) # (batch_size, num_sentences, seq_len, embedding_size)
        sentence_mask = util.get_text_field_mask(sentences)
        # print('Sentence mask:', sentence_mask.size()) # (batch_size, num_sentences, seq_len)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)
        # print('Encoded sentence:', encoded_sentence.size()) # (batch_size, num_sentences, embedding_size)

        logits = self.classifier_feedforward(encoded_sentence) # (batch_size, num_sentences, num_labels(2))
        logits = logits.squeeze(-1) # Actually doesnt do anything (batch_size, num_sentences, num_labels)

        output_dict = {'logits': logits}
        if labels is not None:
            # print("label shape:", labels.shape)
            # print("logits shape:", logits.shape)
            # loss = self.loss(logits, labels.squeeze(-1))
            loss = multiple_target_CrossEntropyLoss(logits, labels)
            # print('Loss multiple_target_CrossEntropyLoss', loss)
            # Get loss using get_loss function
            # sup_batch = copy.deepcopy(sentences)
            # ori_unsup_batch = copy.deepcopy(sentences)
            # aug_unsup_batch = copy.deepcopy(sentences)
            loss, _, _ = self.get_loss(sentences, labels, ori_unsup_sentences, aug_unsup_sentences, 0) # (sup_batch, sup_batch_label, unsup_batch, global_step)
            # print('Loss get_loss function', loss)
            for metric in self.metrics.values():
                metric(logits, labels.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Coverts tag ids to actual tags.
        """
        # for instance_labels in output_dict["logits"]:
            # print('Instance labels:', instance_labels)
        # output_dict["labels"] = [
        #     [self.vocab.get_token_from_index(label, namespace='labels')
        #          for label in instance_labels]
        #         for instance_labels in output_dict["logits"]
        # ]
        output_dict["labels"] = [
            [np.argmax(label.cpu().data.numpy()) for label in instance_labels]
                for instance_labels in output_dict["logits"]
        ]
        # print(output_dict["logits"])
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

# %%
"""Prepare embeddings"""
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(
    pretrained_model = "./biobert_v1.1_pubmed/weights.tar.gz",
    top_layer_only=True,
    requires_grad=False
)

#print('Bert Model:', bert_embedder.bert_model.encoder.layer[11])
for param in bert_embedder.bert_model.encoder.layer[8:].parameters():
    param.requires_grad = True


word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder(
                                                            token_embedders={"tokens": bert_embedder}, 
                                                            allow_unmatched_keys=True)

# %%
BERT_DIM = word_embeddings.get_output_dim()
print('Bert dim:', BERT_DIM)

class BertSentencePooler(Seq2VecEncoder):
    def __init__(self, vocab):
        super().__init__(vocab)

    def forward(self, embs:torch.tensor, mask:torch.tensor=None) -> torch.tensor:
        bert_out = embs[:, :, 0]
        return bert_out
    
    def get_output_dim(self) -> int:
        return BERT_DIM

sentence_encoder = BertSentencePooler(vocab)

# %%
# classifier_feedforward = nn.Linear(256, 2)
classifier_feedforward = nn.Linear(768,2)

# %%
model = BaselineModel(
    vocab,
    word_embeddings,
    sentence_encoder,
    classifier_feedforward
)

# %%
"""Basic sanity check"""
batch = next(iter(iterator(train_dataset)))
# print(batch)
tokens = batch["sentences"]
labels = batch["labels"]

# %%
# mask = util.get_text_field_mask(tokens)

# %%
# embeddings = model.text_field_embedder(tokens)

# %%
# state = model.sentence_encoder(embeddings, mask)

# %%
# logits = model.classifier_feedforward(state)
# logits = logits.squeeze(-1)

# %%
# loss =  nn.NLLLoss()(logits.reshape(-1, 10), labels.reshape(-1, 10))
#def multiple_target_CrossEntropyLoss(logits, labels):
    # loss = 0
    # for i in range(logits.shape[0]):
        # loss = loss + nn.CrossEntropyLoss(weight=torch.tensor([1,3]))(logits[i, :, :], labels[i, :])
    # return loss

# %%
# loss.backward()

# %%
# loss = model(**batch)["loss"]

# %%
"""Train"""
# print('Parameters:', model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer)
optimizer = optim.SGD([{'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[11].parameters(), 'lr': 0.001},
                        {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[10].parameters(), 'lr': 0.00095},
                        # {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[9].parameters(), 'lr': 0.0009},
                        # {'params': model.text_field_embedder.token_embedder_tokens.bert_model.encoder.layer[8].parameters(), 'lr': 0.000855}
                        ], lr=0.001)
# Default
# optimizer = optim.SGD(model.parameters(), lr=0.001)

model = model.cuda()

print('Start training')

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    validation_iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    patience=3,
    num_epochs=50,
    cuda_device=[0, 1]
)

# %%
metrics = trainer.train()

# %%
"""Testing"""

class ClaimCrfPredictor(Predictor):
    """
    Predictor wrapper for the AcademicPaperClassifier
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentences = json_dict['sentences']
        instance = self._dataset_reader.text_to_instance(sents=sentences)
        return instance

def read_json(file_path):
    """
    Read list from JSON path
    """
    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as fp:
            ls = [json.loads(line) for line in fp]
        return ls

# %%
test_list = read_json(cached_path(VALIDATION_PATH))
claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
y_pred, y_true = [], []
for tst in validation_dataset:
    # print('tst', tst)
    pred = claim_predictor.predict_instance(tst)
    # print('Pred output:', pred)
    logits = torch.FloatTensor(pred['logits'])
    # print(logits.shape)
    # print('Logits output:', logits)
#     best_paths = model.crf.viterbi_tags(torch.FloatTensor(pred['logits']).unsqueeze(0),
#                                         torch.LongTensor(pred['mask']).unsqueeze(0))
    predicted_labels = pred['labels']
    y_pred.extend(predicted_labels)
    y_true.extend(tst['labels'])
    # break
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)
# print('Y true:', y_true)
# print('Y pred:', y_pred)
print('Val score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))
# Save y_true and y_pred
df = pd.DataFrame()
df['y_true'] = y_true
df['y_pred'] = y_pred
df.to_csv('biobert_y_true_pred_val.csv', index=False)

# %%
test_list = read_json(cached_path(TEST_PATH))
claim_predictor = ClaimCrfPredictor(model, dataset_reader=reader)
y_pred, y_true = [], []
for tst in test_dataset:
    pred = claim_predictor.predict_instance(tst)
    logits = torch.FloatTensor(pred['logits'])
    predicted_labels = pred['labels']
    y_pred.extend(predicted_labels)
    y_true.extend(tst['labels'])
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)
print('Test score:', precision_recall_fscore_support(y_true, y_pred, average='binary'))

# Save model
with open(f"./finetune_model.th", "wb") as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files(f"./finetune_vocab.txt")

# Save y_true and y_pred
df = pd.DataFrame()
df['y_true'] = y_true
df['y_pred'] = y_pred
df.to_csv('biobert_y_true_pred_test.csv', index=False)