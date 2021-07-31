"""
This file train a customized bert model based on the cleaning train text
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW
import torch.nn as nn


from Bert_Arch import Bert_Arch

def create_dataloader(data_seq, data_mask, data_y):
    """
    create dataloader for pytorch
    Args:
        data_seq: [torch tensor] tokens id in bert tokenizer
        data_mask: [torch tensor] mask id for bert input
        data_y: [torch tensor] categorical label as output

    Returns:
        dataloader: dataloader for training

    """
    BATCH_SIZE = 32

    data = TensorDataset(data_seq,data_mask, data_y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler,batch_size=BATCH_SIZE)

    return dataloader

def train(train_dataloader,model,cross_entropy):
    """
    train the model
    Args:
        train_dataloader: train dataset
        model: training network
        cross_entropy: loss function to back propagation

    Returns:
        avg_loss: average loss during learning process
        total_preds: totally prediction from network
    """


    optimizer = AdamW(model.parameters(),lr=1e-3)
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print(' Batch{:>5} of {:>5,}.'.format(step,len(train_dataloader)))

        sent_id, mask, label = batch
        model.zero_grad()
        preds = model(sent_id,mask)
        loss = cross_entropy(preds,label)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
        optimizer.step()
        preds = preds.detach().numpy()
        total_preds.append(preds)
    avg_loss = total_loss/len(train_dataloader)
    total_preds = np.concatenate(total_preds,axis=0)

    return avg_loss, total_preds


def evaluate(val_dataloader,model,cross_entropy):
    """
    evaluate current model at the end of each epoch
    Args:
        val_dataloader: dataset for evaluation
        model: current trained model
        cross_entropy: loss function

    Returns:
        avg_loss: average loss during learning process
        total_preds: totally prediction from network
    """
    print('\n Evaluating...')
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            print(' Batch{:>5} of {:>5,}.'.format(step, len(val_dataloader)))
        sent_id, mask, label = batch
        with torch.no_grad():
            preds = model(sent_id,mask)
            loss = cross_entropy(preds,label)
            total_loss += loss.item()
            preds = preds.detach().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(preds, axis=0)

    return avg_loss, total_preds


if __name__=='__main__':
    MAX_LENGTH = 50
    EPOCHS = 100
    path_train = 'dataset_for_shared_task\\dataset_for_shared_task\\train_clean_1.csv'

    df_train = pd.read_csv(path_train, sep=',')

    train_text, val_text, train_label, val_label = train_test_split(df_train['text'],df_train['label'],
                                                                        test_size=0.3,
                                                                        stratify=df_train['label']
                                                                        )
    seq_len = [len(i.split()) for i in train_text]
    pd.Series(seq_len).hist(bins=30)

    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_label.tolist())
    train_y = train_y.type(torch.LongTensor)

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_label.tolist())
    val_y = val_y.type(torch.LongTensor)

    train_dataloader = create_dataloader(train_seq, train_mask, train_y)
    val_dataloader = create_dataloader(val_seq,val_mask,val_y)

    for param in bert.parameters():
        param.requires_grad = False
    model = Bert_Arch(bert)
    class_weights = compute_class_weight('balanced', np.unique(train_label), train_label)
    weights = torch.tensor(class_weights, dtype=torch.float)
    cross_entropy = nn.NLLLoss(weight=weights)

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    for epoch in range(EPOCHS):
        print('\n Epoch{:}/{:}'.format(epoch+1,EPOCHS))
        train_loss,_ = train(train_dataloader,model,cross_entropy)
        valid_loss,_ = evaluate(val_dataloader,model,cross_entropy)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),'save_weights.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')










