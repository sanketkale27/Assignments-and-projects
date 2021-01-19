""" A neural chatbot using sequence to sequence model with
attentional decoder. 
This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
Sequence to sequence model by Cho et al.(2014)
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
This file contains the hyperparameters for the model.
See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
#DATA_PATH = 'D:/Study/NLP/assig2/Baseline_ChatBot/Baseline_ChatBot/chat_corpus-master/chat_corpus-master/chat.txt'
DATA_PATH = 'D:/Study/NLP/assig2/Baseline_ChatBot/Baseline_ChatBot/chat_corpus-master/chat_corpus-master/chat.txt'
DATA_PATH_MS = 'D:/Study/NLP/assig2/Baseline_ChatBot/Baseline_ChatBot/chat_corpus-master/chat_corpus-master/movie_subtitles_en.txt'

CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'
OUTPUT_FILE_Joey = 'output_convo_joey.txt'
OUTPUT_FILE_F = 'feedback.txt'
#PROCESSED_PATH_MS = 'D:/Study/NLP/assig2/Baseline_ChatBot/Baseline_ChatBot/chat_corpus-master/chat_corpus-master/processed'
#CPT_PATH_MS = 'D:/Study/NLP/assig2/Baseline_ChatBot/Baseline_ChatBot/chat_corpus-master/chat_corpus-master/checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 2500

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63),(32, 285)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

# Thanks to https://stackoverflow.com/a/43023503/3971619
contractions_new = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he shall have / he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 24484
DEC_VOCAB = 24660
ENC_VOCAB = 34771
DEC_VOCAB = 30527
ENC_VOCAB = 48588
DEC_VOCAB = 43257
ENC_VOCAB = 51881
DEC_VOCAB = 45091
ENC_VOCAB = 39381
DEC_VOCAB = 34583
ENC_VOCAB = 39427
DEC_VOCAB = 34584
ENC_VOCAB = 39459
DEC_VOCAB = 34592
ENC_VOCAB = 688
DEC_VOCAB = 845
ENC_VOCAB = 41327
DEC_VOCAB = 36247
ENC_VOCAB = 41329
DEC_VOCAB = 36243
ENC_VOCAB = 660
DEC_VOCAB = 834
ENC_VOCAB = 39181
DEC_VOCAB = 34671
ENC_VOCAB = 39181
DEC_VOCAB = 34671
ENC_VOCAB = 16429
DEC_VOCAB = 17521
ENC_VOCAB = 16436
DEC_VOCAB = 17519
ENC_VOCAB = 16450
DEC_VOCAB = 17535
ENC_VOCAB = 39185
DEC_VOCAB = 34645
ENC_VOCAB = 35275
DEC_VOCAB = 35297
ENC_VOCAB = 35235
DEC_VOCAB = 35290
