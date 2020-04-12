# %%
from typing import Dict, List
from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, SequenceLabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer


@DatasetReader.register("crf_pubmed_rct")
class CrfPubmedRCTReader(DatasetReader):
    """
    Reads a file from Pubmed RCT text file.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as file:
            sents = []
            labels = []
            for line in file:
                if not line.startswith('#') and line.strip() != '' and '\t' in line:
                    label, sent = line.split('\t')
                    sents.append(sent.strip())
                    labels.append(label)
                elif len(sents) > 0 and len(labels) > 0:
                    yield self.text_to_instance(sents, labels)
                    sents = []
                    labels = []
                else:
                    continue

    @overrides
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
train_data_path = "train.txt"
validation_data_path = "dev.txt"
test_data_path = "test.txt"

# %%
reader = CrfPubmedRCTReader()
# test_dataset = reader.read(validation_data_path)
# test_dataset = reader.read(test_data_path)
test_dataset = reader.read(train_data_path)

# %%
'''Write to new file'''
# example_file = open('new_dev.txt', 'w')
# example_file = open('new_test.txt', 'w')
example_file = open('new_train.txt', 'w')

for data in test_dataset:
    sentences = data['sentences']
    for sent in sentences:
        sentence = ''
        for word in list(sent):
            sentence += word.text
            sentence += ' '
        example_file.write(sentence)
        example_file.write('\n')
    example_file.write('\n\n\n')

example_file.close()
# %%
