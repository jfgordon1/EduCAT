import sys
sys.path.append('..')
import CAT
import json
import logging
import numpy as np
import pandas as pd

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
setuplogger()

# choose dataset here
dataset = 'assistment'
# modify config here
config = {
    'learning_rate': 0.002,
    'batch_size': 2048,
    'num_epochs': 10,
    'num_dim': 1, # for IRT or MIRT
    'device': 'cpu',
    # for NeuralCD
    'prednet_len1': 128,
    'prednet_len2': 64,
    'betas': (0.9, 0.999),
}

# read datasets
train_triplets = pd.read_csv(f'../data/{dataset}/train_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../data/{dataset}/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'../data/{dataset}/metadata.json', 'r'))

train_data = CAT.dataset.TrainDataset(train_triplets, concept_map,
                                      metadata['num_train_students'], 
                                      metadata['num_questions'], 
                                      metadata['num_concepts'])

# define model here
model = CAT.model.IRTModel(**config)
# train model
model.init_model(train_data)
model.train(train_data, log_step=10)

# save model
model.adaptest_save('../ckpt/irt.pt')