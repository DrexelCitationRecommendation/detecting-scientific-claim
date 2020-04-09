import json
import os

# print(os.listdir())
file_path = './train_labels.json'
preprocessed_file_path = './preprocessed_train_labels.txt'

preprocessed_file = open(preprocessed_file_path, "w")

with open(file_path, 'r') as file:
    for line in file:
        # print(line)
        example = json.loads(line)
        sents = example['sentences']
        labels = example['labels']
        for sent in sents:
            if '\n' in sent:
                sent = sent.replace('\n', ' ')
                preprocessed_file.write(sent)
                preprocessed_file.write('\n')
                pass
            elif sent == '.':
                # augmented_sent = ['.'] * 10
                # print('sents', sents)
                # print('len sents', len(sents))
                # print('labels', labels)
                # print('len labels', len(labels))
                preprocessed_file.write(sent)
                preprocessed_file.write('\n')
            else:
                preprocessed_file.write(sent)
                preprocessed_file.write('\n')
        preprocessed_file.write('\n\n\n')

preprocessed_file.close()