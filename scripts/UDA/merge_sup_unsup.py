# %%
import json
import os
import random

random.seed(42)

# %%
uda_ratio = 3
sup_train_data_path = './train_labels.json'
original_unsup_data_path = './new_dev.txt'
augmented_unsup_data_path = './back_trans_data_remove_tag_dev/paraphrase/file_0_of_1.json'
combined_train_data_path = './combined_sup_train_data_unsup_dev_pubmed_data.json' # Combined file with no 0-claim abstract

sup_train_data_sentences = []
sup_train_data_labels = []
ori_unsup_data_sentences = []
aug_unsup_data_sentences = []

combined_train_data_abstracts = []

# %%
def ratio_0_and_1(lst):
    count_0 = 0
    count_1 = 0
    for i in lst:
        if i == '0':
            count_0 += 1
        elif i == '1':
            count_1 += 1
    return count_1 / count_0

def sum_str(lst):
    total = 0
    for i in lst:
        total += int(i)
    return total

# %%
count_ratio_smaller = 0
count_0_claim = 0
with open(sup_train_data_path, 'r') as file:
    for line in file:
        example = json.loads(line)
        sents = example['sentences']
        labels = example['labels']
        if ratio_0_and_1(labels) < 1/10: # 1/10 has 80 abstracts that has 0 claim
            count_ratio_smaller += 1
            sup_train_data_sentences.append(sents)
            sup_train_data_labels.append(labels)
        else:
            sup_train_data_sentences.append(sents)
            sup_train_data_labels.append(labels)

            if len(sents) != len(labels):
                print('Not match sents and labels')

with open(original_unsup_data_path, 'r') as file:
    sents = []
    for line in file:
        if line.strip() != '':
            sents.append(line.strip())
        elif len(sents) > 0:
            ori_unsup_data_sentences.append(sents)
            sents = []
        else:
            continue

with open(augmented_unsup_data_path, 'r') as file:
    sents = []
    for line in file:
        if line.strip() != '':
            sents.append(line.strip())
        elif len(sents) > 0:
            aug_unsup_data_sentences.append(sents)
            sents = []
        else:
            continue

print('Sup data size:', len(sup_train_data_sentences))
print('Ori unsup data size:', len(ori_unsup_data_sentences))
print('Aug unsup data size:', len(aug_unsup_data_sentences))

# %%
count_passed = 0
# Create new combined file
for i in range(len(sup_train_data_sentences)):
    count_passed += 1
    for t in range(uda_ratio):
        abstract = {}
        abstract['sentences'] = sup_train_data_sentences[i]
        abstract['labels'] = sup_train_data_labels[i]
        abstract['ori_unsup_sentences'] = ori_unsup_data_sentences[i * uda_ratio + t]
        abstract['aug_unsup_sentences'] = aug_unsup_data_sentences[i * uda_ratio + t]
        
        combined_train_data_abstracts.append(abstract)

print('Number abstracts passed:', count_passed)

# %%
print('Total length combined:', len(combined_train_data_abstracts))
random.shuffle(combined_train_data_abstracts)
with open(combined_train_data_path, 'w') as outfile:
    for abstract in combined_train_data_abstracts:
        json.dump(abstract, outfile)
        outfile.write('\n')