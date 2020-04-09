# %%
import json
import os
import random

random.seed(42)

# %%
original_train_data_path = './train_labels.json'
augmented_train_data_path = './back_trans_data_preprocessed_claim_train/paraphrase/file_0_of_1.json'
combined_train_data_path = './combined_back_translate_train_labels.json'

original_train_data_labels = []
original_train_data_sentences = []
augmented_train_data_sentences = []

combined_train_data_abstracts = []

with open(original_train_data_path, 'r') as file:
    for line in file:
        example = json.loads(line)
        sents = example['sentences']
        labels = example['labels']
        original_train_data_sentences.append(sents)
        original_train_data_labels.append(labels)
        if len(sents) != len(labels):
            print('Not match sents and labels')

with open(augmented_train_data_path, 'r') as file:
    sents = []
    for line in file:
        if line.strip() != '':
            sents.append(line.strip())
        elif len(sents) > 0 and len(labels) > 0:
            augmented_train_data_sentences.append(sents)
            sents = []
        else:
            continue

# %%
# Create new combined file
for i in range(len(original_train_data_sentences)):
    original_abstract = {}
    original_abstract['sentences'] = original_train_data_sentences[i]
    original_abstract['labels'] = original_train_data_labels[i]

    augmented_abstract = {}
    augmented_abstract['sentences'] = augmented_train_data_sentences[i]
    augmented_abstract['labels'] = original_train_data_labels[i]

    combined_train_data_abstracts.append(original_abstract)
    combined_train_data_abstracts.append(augmented_abstract)

# %%
# print(len(combined_train_data_abstracts))
random.shuffle(combined_train_data_abstracts)
with open(combined_train_data_path, 'w') as outfile:
    for abstract in combined_train_data_abstracts:
        json.dump(abstract, outfile)
        outfile.write('\n')