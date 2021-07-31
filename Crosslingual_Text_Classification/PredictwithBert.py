"""
evaluate on test set
"""
import pandas as pd
from transformers import AutoModel, BertTokenizerFast
import torch
import numpy as np
from sklearn.metrics import classification_report

from CleanText import CleanText
from Bert_Arch import Bert_Arch

"""path1 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\train_annotators_1.csv'
path2 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\train_annotators_2.csv'
path3 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\test_annotators_1.csv'
path4 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\test_annotators_2.csv'
path5 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\test_annotators_11.csv'
path6 = 'dataset_for_shared_task\\dataset_for_shared_task\\test\\test_annotators_21.csv'
df_test1 = pd.read_csv(path1,sep=',')
df_test2 = pd.read_csv(path2,sep=',')
df_test3 = pd.read_csv(path3,sep=',')
df_test4 = pd.read_csv(path4,sep=',')
df_test3_l = pd.read_csv(path5,sep=',')
df_test4_l = pd.read_csv(path6,sep=',')
df_test3['label'] = df_test3_l['label']
df_test4['label'] = df_test4_l['label']
df_temp = pd.concat([df_test1,df_test2,df_test3,df_test4],axis=0)
df_test = pd.DataFrame(columns=['text','label'])
df_test['text'] = df_temp['text']
df_test['label'] = df_temp['label']
df_test.to_csv('testset_new.csv',index=False)"""

"""df_test = pd.read_csv(path_test, sep=',')
df_test = df_test[['text', 'label']]
X_test = df_test['text'].tolist()
y_test = df_test['label']
y_test = y_test.replace(3, 2)
ct = CleanText(X_test)
X_test_clean = ct.clean_text()
df_test_tocsv = pd.DataFrame(columns=['text','label'])
df_test_tocsv['text'] = X_test_clean
df_test_tocsv['label'] = y_test
df_test_tocsv.to_csv('test_clean.csv',index=False)"""


path_test = 'dataset_for_shared_task\\dataset_for_shared_task\\test_clean.csv'
path_weight = 'save_weights.pt'
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
MAX_LENGTH = 50
bert = AutoModel.from_pretrained('bert-base-uncased')

df_test = pd.read_csv(path_test, sep=',')
test_text = df_test['text'].values
test_label = df_test['label'].values

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
    )
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_label.tolist())
test_y = test_y.type(torch.LongTensor)

model = Bert_Arch(bert)
model.load_state_dict(torch.load(path_weight))
with torch.no_grad():
    preds = model(test_seq,test_mask)
    preds.detach().numpy()
preds = np.argmax(preds,axis=1)
print(classification_report(test_y,preds))