# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: pierrou_env
#     language: python
#     name: pierrou_env
# ---

# # 1 - NER PyTorch
#
#
# **Sources** :
# - Source de données Kaggle : https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus
# - Tuto Abishek : https://www.youtube.com/watch?v=MqQ7rqRllIc

# +
# %load_ext autoreload
# %autoreload 2

import os
os.chdir('..')

import joblib
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ner_pytorch.config.params import PARAMS
from ner_pytorch.dataset import EntityDataset
from ner_pytorch.engine import train_fn, eval_fn
from ner_pytorch.model import EntityModel
from ner_pytorch.utils import *
from ner_pytorch.preprocessing import process_data

# +
data = pd.read_csv(PARAMS.PATHS.TRAIN, encoding='latin-1')
data["Sentence #"] = data["Sentence #"].fillna(method='ffill')

num_tag = data.Tag.nunique()
num_pos = data.POS.nunique()
print(f'POS {num_pos} categories :', data.POS.unique())
print(f'Tag {num_tag} categories :', data.Tag.unique(), end='\n\n')

data.shape
data.head(7)
# -

sentences, pos, tag, label_enc_POS, label_enc_NER = process_data(data)
joblib.dump(label_enc_POS, 'data/outputs/label_enc_POS.joblib')
joblib.dump(label_enc_NER, 'data/outputs/label_enc_NER.joblib')

# Démo : 
i = 6
print(sentences[i])
print(pos[i])
print(tag[i])
# label_enc_POS.classes_
# label_enc_NER.classes_

# On split notre jeu de données de la façon suivante :
# - `test` = 20%
# - `train` = [`minitrain`, `valid`] = 80%
# - `minitrain` = 60%
# - `valid` = 20%

# +
len_test = int(PARAMS.SAMPLE_SIZES.TEST * len(sentences))
len_valid = int(PARAMS.SAMPLE_SIZES.VALID * len(sentences))

(
    sentences_train, sentences_test,
    pos_train, pos_test,
    tag_train, tag_test
) = train_test_split(sentences, pos, tag, random_state=PARAMS.SEED, 
                     test_size=len_test)

(
    sentences_minitrain, sentences_valid,
    pos_minitrain, pos_valid,
    tag_minitrain, tag_valid
) = train_test_split(sentences_train, pos_train, tag_train, 
                     random_state=PARAMS.SEED, test_size=len_valid)

len(sentences_minitrain), len(sentences_valid), len(sentences_test)

# +
minitrain_dataset = EntityDataset(
    texts=sentences_minitrain, pos=pos_minitrain, tags=tag_minitrain
)
minitrain_data_loader = DataLoader(
    minitrain_dataset, batch_size=PARAMS.MODEL.TRAIN_BATCH_SIZE, num_workers=2 
)

valid_dataset = EntityDataset(
    texts=sentences_valid, pos=pos_valid, tags=tag_valid
)
valid_data_loader = DataLoader(
    valid_dataset, batch_size=PARAMS.MODEL.VALID_BATCH_SIZE, num_workers=1
)

test_dataset = EntityDataset(
    texts=sentences_test, pos=pos_test, tags=tag_test
)
test_data_loader = DataLoader(
    test_dataset, batch_size=PARAMS.MODEL.VALID_BATCH_SIZE, num_workers=1
)
# -

i = 2
print(test_dataset.texts[i])
for key, value in valid_dataset[i].items():
    print(key + ':', value)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

model = EntityModel(num_tag=num_tag, num_pos=num_pos)
model.to(device);

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

# +
num_train_steps = int(len(sentences_minitrain) / PARAMS.MODEL.TRAIN_BATCH_SIZE * PARAMS.MODEL.EPOCHS)
num_train_steps # nb de fois que l'on va envoyer un batch dans le réseau

optimizer = torch.optim.AdamW(optimizer_parameters, lr=PARAMS.MODEL.LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)
# -

reprendre ici

for temp in minitrain_data_loader:
    break

temp

best_loss = np.inf
for epoch in range(PARAMS.MODEL.EPOCHS):
    train_loss = train_fn(minitrain_data_loader, model, optimizer, device, scheduler)
    test_loss = eval_fn(valid_data_loader, model, device)
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        # torch.save(model.state_dict(), PARAMS.PATHS.MODEL_SAVED)
        best_loss = test_loss


