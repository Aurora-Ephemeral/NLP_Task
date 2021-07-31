"""
Semantic Similarity as classification problem with single Bert. This script is to generate negative
label and split the whole dataset into train and dev set
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

data_path = "../data/SharedTaskFinal/training.tsv"
# generate pos data set
df_data_pos = pd.read_csv(data_path, sep='\t', encoding='utf-8')
samples,_ = df_data_pos.shape
labels = [1 for i in range(samples)]
df_data_pos.insert(2, "label", labels, True)
# generate neg data set
df_data_neg = pd.DataFrame(columns=['query','clarifying_question','label'])
unique_query = df_data_pos["query"].unique()
sents_num = unique_query.shape[0]
# randomly select clarify question from other sentence pair
model = SentenceTransformer('stsb-roberta-large')
for i in range(sents_num):
    this_query = unique_query[i]
    num_pos_pair = df_data_pos[df_data_pos['query'] == this_query].count().values[0]
    df_neg_candidate = df_data_pos[df_data_pos['query'] != this_query]['clarifying_question']
    #df_neg_select = df_neg_candidate.sample(n=num_pos_pair)['clarifying_question']
    embedding1 = model.encode(sentences=[this_query],convert_to_tensor=True)
    embedding2 = model.encode(sentences=df_neg_candidate.tolist(), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

    df_neg_select = pd.DataFrame(df_neg_candidate)
    df_neg_select.insert(0, 'query', this_query,)
    df_neg_select.insert(2,'label', 0)
    df_data_neg = pd.concat((df_data_neg,df_neg_select), axis=0)



# split the data to train and dev set

train_pos, dev_pos = train_test_split(df_data_pos, train_size=0.8)
train_neg, dev_neg = train_test_split(df_data_neg, train_size=0.8)

train_set = pd.concat((train_pos, train_neg), axis=0)
dev_set = pd.concat((dev_pos, dev_neg), axis=0)

train_set = train_set.sample(frac=1.0).reset_index(drop=True)
dev_set = dev_set.sample(frac=1.0).reset_index(drop=True)

train_set.to_csv("../data/train_set_whole.csv", index=False)
dev_set = dev_set.to_csv("../data/dev_set_whole.csv", index=False)













