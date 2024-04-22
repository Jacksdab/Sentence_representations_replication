import torch
import torchtext
import nltk
import pickle
import time
from nltk.tokenize import word_tokenize
from torchtext import datasets, vocab
from torchtext.data import Field, Dataset, BucketIterator

# Define a tokenizer function using spaCy
def tokenize_en(text):
    return word_tokenize(text)

def tokenize(s):
    return s.split(' ')

def data_preprocessing(batch_size = 64, root = '.data', debug = False, device = None):
    if debug == True:
        root = '.debug_data'

    TEXT = Field(sequential=True, tokenize = tokenize_en, lower = True, batch_first = True)
    LABEL = Field(sequential = False, is_target = True, batch_first = True)
    start_time = time.time()
    train_set, val_set, test_set = datasets.SNLI.splits(text_field = TEXT, label_field= LABEL, root = root)
    split_time = time.time() - start_time

    nltk.download('punkt')

    print(f"split retrieved. it took {split_time}")

    # build vocab & get glove 
    glove_embedding = vocab.GloVe(name='840B', dim=300)
    TEXT.build_vocab(train_set, vectors = glove_embedding)
    LABEL.build_vocab(train_set)
    print("Getting vocab...")
    vocabulary = TEXT.vocab

    # get datasets batch iterators
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train_set, val_set, test_set),
            batch_size= batch_size,
            sort_within_batch=True,
            device= device)
    
    return vocabulary, train_iter, val_iter, test_iter

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        iterators = pickle.load(f)
    return iterators

# vocabulary, train_iter, val_iter, test_iter = data_preprocessing()

# # train_iter_list = list(train_iter)
# # val_iter_list = list(val_iter)
# # test_iter_list = list(test_iter)



# # save_pickle((train_iter_list, val_iter_list, test_iter_list), 'iterators.pkl')
# for batch in train_iter:
#     # Access the first sample in the batch
#     first_premise = batch.premise[0]  # Accessing the first premise in the batch
#     first_hypothesis = batch.hypothesis[0]  # Accessing the first hypothesis in the batch
    
#     # Print or inspect the first samples
#     print("First premise:", first_premise)
#     print("First hypothesis:", first_hypothesis)
    
#     # Break after inspecting the first batch if you want to see just one batch
#     break
# # Get the first sample from the batch
# print("Unique labels:", torch.unique(train_iter.label))
# # Print the sample attributes
