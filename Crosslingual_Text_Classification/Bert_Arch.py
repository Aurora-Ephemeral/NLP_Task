"""
This file define customized bert architecture
"""
import torch.nn as nn

class Bert_Arch(nn.Module):

    def __init__(self,bert):

        super(Bert_Arch,self).__init__()

        self.bert = bert

        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

        self.elu = nn.ELU()

        self.fc1 = nn.Linear(768,512)

        self.fc2 = nn.Linear(512,256)

        self.fc3 = nn.Linear(256,3)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id,mask):

        _, cls_hs = self.bert(sent_id, attention_mask=mask)


        x = self.fc1(cls_hs)

        #x = self.dropout(x)

        x = self.sigmoid(x)

        #x = self.dropout(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        x = self.fc3(x)

        x=self.softmax(x)

        return x