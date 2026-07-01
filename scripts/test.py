import sys
sys.path.append('..')
import CAT
import json
import torch
import logging
import datetime
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    
setuplogger()

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

log_dir = f"../logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}/"
log_dir = f"../logs/"
print(log_dir)
writer = SummaryWriter(log_dir)

# choose dataset here
import CAT.strategy


dataset = 'assistment'
# modify config here
config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 8,
    'num_dim': 1, # for IRT or MIRT
    'device': 'cpu',
    # for NeuralCD
    'prednet_len1': 128,
    'prednet_len2': 64,
    # for BOBCAT
    'policy':'notbobcat',
    'betas': (0.9, 0.999),
    'policy_path': 'policy.pt',
    # for NCAT
    'THRESHOLD' :300,
    'start':0,
    'end':3000
    
}
# fixed test length
test_length = 5
# choose strategies here
#strategies = [CAT.strategy.RandomStrategy(), CAT.strategy.MFIStrategy(), CAT.strategy.KLIStrategy()]
strategies = [CAT.strategy.NCATs()]
# modify checkpoint path here
ckpt_path = '../ckpt/irt.pt'
bobcat_policy_path =config['policy_path']

test_triplets = pd.read_csv(f'../data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../data/{dataset}/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'../data/{dataset}/metadata.json', 'r'))

test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                        metadata['num_test_students'], 
                                        metadata['num_questions'], 
                                        metadata['num_concepts'])

import warnings
warnings.filterwarnings("ignore")

for strategy in strategies:
    avg =[]
    model = CAT.model.IRTModel(**config)
    #model = CAT.model.NCDModel(**config)
    model.init_model(test_data)
    model.adaptest_load(ckpt_path)
    test_data.reset()
    print(strategy.name)
    if strategy.name == 'NCAT':
        selected_questions = strategy.adaptest_select(test_data,concept_map,config,test_length)
        for it in range(test_length):
            for student, questions in selected_questions.items():
                test_data.apply_selection(student, questions[it])  
            model.adaptest_update(test_data)
            results = model.evaluate(test_data)
        # log results
            logging.info(f'Iteration {it}')
            for name, value in results.items():
                logging.info(f'{name}:{value}')
        continue
    if strategy.name == 'BOBCAT':
        real = {}
        real_data = test_data.data
        for sid in real_data:
            question_ids = list(real_data[sid].keys())
            real[sid]={}
            tmp={}
            for qid in question_ids:
                tmp[qid]=real_data[sid][qid]
            real[sid]=tmp
    logging.info('-----------')
    logging.info(f'start adaptive testing with {strategy.name} strategy')
    logging.info(f'Iteration 0')
    # evaluate models
    results = model.evaluate(test_data)
    for name, value in results.items():
        logging.info(f'{name}:{value}')
    S_sel ={}
    for sid in range(test_data.num_students):
        key = sid
        S_sel[key] = []
    selected_questions={}
    for it in range(1, test_length + 1):
        logging.info(f'Iteration {it}')
        # select question
        if strategy.name == 'BOBCAT':
            selected_questions = strategy.adaptest_select(model, test_data,S_sel)
            for sid in range(test_data.num_students):
                tmp = {}
                tmp[selected_questions[sid]] = real[sid][selected_questions[sid]]
                S_sel[sid].append(tmp)
        elif it == 1 and strategy.name == 'BECAT Strategy':
            for sid in range(test_data.num_students):
                untested_questions = np.array(list(test_data.untested[sid]))
                random_index = random.randint(0, len(untested_questions)-1)
                selected_questions[sid] = untested_questions[random_index]
                S_sel[sid].append(untested_questions[random_index])
        elif strategy.name == 'BECAT Strategy':    
            selected_questions = strategy.adaptest_select(model, test_data,S_sel)
            for sid in range(test_data.num_students):
                S_sel[sid].append(selected_questions[sid])
        else:
            selected_questions = strategy.adaptest_select(model, test_data)
        for student, question in selected_questions.items():
            test_data.apply_selection(student, question)       
        
        # update models
        model.adaptest_update(test_data)
        # evaluate models
        results = model.evaluate(test_data)
        # log results
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, it)
            
