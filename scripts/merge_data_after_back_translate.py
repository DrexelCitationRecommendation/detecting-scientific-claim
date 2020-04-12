# %%
import json
import os
import random

random.seed(42)

# %%
original_train_data_path = './train_labels.json'
augmented_train_data_path = './back_trans_data_preprocessed_claim_train/paraphrase/file_0_of_1.json'
augmented_train_data_path_2 = './back_trans_data_preprocessed_claim_train_2/paraphrase/file_0_of_1.json'
# combined_train_data_path = './combined_back_translate_train_labels.json' # Original combined file
combined_train_data_path = './combined_back_translate_higher_1_8_train_labels.json' # Combined file with no 0-claim abstract

original_train_data_labels = []
original_train_data_sentences = []
augmented_train_data_sentences = []
augmented_train_data_sentences_2 = []

combined_train_data_abstracts = []

count_0 = 0
count_1 = 0
count_ratio_smaller = 0

def sum_str(lst):
    total = 0
    for i in lst:
        total += int(i)
    return total

def ratio_0_and_1(lst):
    count_0 = 0
    count_1 = 0
    for i in lst:
        if i == '0':
            count_0 += 1
        elif i == '1':
            count_1 += 1
    return count_1 / count_0


with open(original_train_data_path, 'r') as file:
    for line in file:
        example = json.loads(line)
        sents = example['sentences']
        labels = example['labels']
        if ratio_0_and_1(labels) < 1/8: # 1/10 has 80 abstracts that has 0 claim
            count_ratio_smaller += 1

        original_train_data_sentences.append(sents)
        original_train_data_labels.append(labels)

        if len(sents) != len(labels):
            print('Not match sents and labels')

with open(augmented_train_data_path, 'r') as file:
    sents = []
    for line in file:
        if line.strip() != '':
            sents.append(line.strip())
        elif len(sents) > 0:
            augmented_train_data_sentences.append(sents)
            sents = []
        else:
            continue

with open(augmented_train_data_path_2, 'r') as file:
    sents = []
    for line in file:
        if line.strip() != '':
            sents.append(line.strip())
        elif len(sents) > 0:
            augmented_train_data_sentences_2.append(sents)
            sents = []
        else:
            continue

# %%
count_passed = 0
# Create new combined file
for i in range(len(original_train_data_sentences)):
    if ratio_0_and_1(original_train_data_labels[i]) < 1/8: # 1/10 has 80 abstracts that has 0 claim
        pass
    else:
        count_passed += 1
        original_abstract = {}
        original_abstract['sentences'] = original_train_data_sentences[i]
        original_abstract['labels'] = original_train_data_labels[i]

        augmented_abstract = {}
        augmented_abstract['sentences'] = augmented_train_data_sentences[i]
        augmented_abstract['labels'] = original_train_data_labels[i]

        augmented_abstract_2 = {}
        augmented_abstract_2['sentences'] = augmented_train_data_sentences_2[i]
        augmented_abstract_2['labels'] = original_train_data_labels[i]

        combined_train_data_abstracts.append(original_abstract)
        combined_train_data_abstracts.append(augmented_abstract)
        combined_train_data_abstracts.append(augmented_abstract_2)

print('Number abstracts passed:', count_passed)

# %%
# print(len(combined_train_data_abstracts))
random.shuffle(combined_train_data_abstracts)
with open(combined_train_data_path, 'w') as outfile:
    for abstract in combined_train_data_abstracts:
        json.dump(abstract, outfile)
        outfile.write('\n')