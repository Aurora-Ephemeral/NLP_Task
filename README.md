# Background
This folder contains different tasks with the topic of NLP under lecture Application of Deeplearning in NLP from TU Darmstadt

# Introduction
Totally it contains two kinds of topics. One is to classify unstructured tweets from EU Parliament into three classes: solidarity, unsolidarity and ambiguous. The other is ranking the semantic similarity among different query and clarifying question
## Cross Lingual Tweets Classification
- Model: Bert + NN
- Preprocessing: remove stop words, punctuation, stemming, translation Germany to English backtranslation between English and Germany
## Semantic Similarity Calculation
- Model: Bert + BiLSTM
- Preprocessing: negative sample generation by random sampling from rest dataset
# Usage
Clone the repo """ git clone https://github.com/Aurora-Ephemeral/NLP_Task.git """


