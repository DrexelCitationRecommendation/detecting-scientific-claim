# %%
import os
import pandas as pd
import numpy as np

# %%
biobert_y_true_pred_test = pd.read_csv('./biobert_y_true_pred_test.csv')
crf_y_true_pred_test = pd.read_csv('./crf_y_true_pred_test.csv')

# %%
# Make sure all the rows are matched in both files
for i, row in biobert_y_true_pred_test.iterrows():
    if row['y_true'] != crf_y_true_pred_test.iloc[i]['y_true']:
        print(i)

# %%
# Count c10
c01 = 0
for i, row in crf_y_true_pred_test.iterrows():
    if row['y_true'] != row['y_pred']:
        if biobert_y_true_pred_test.iloc[i]['y_true'] == biobert_y_true_pred_test.iloc[i]['y_pred']:
            c01 += 1
c10 = 0
for i, row in biobert_y_true_pred_test.iterrows():
    if row['y_true'] != row['y_pred']:
        if crf_y_true_pred_test.iloc[i]['y_true'] == crf_y_true_pred_test.iloc[i]['y_pred']:
            c10 += 1
        
# %%
mcnemar_calculation = (abs(c01 - c10) - 1)**2 / (c10 + c01)

# %%
print('c01:', c01)
print('c10:', c10)
print('McNemar result:', mcnemar_calculation)

# %%
