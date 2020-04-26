
import pandas as pd
import json
import numpy as np
import csv
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Protein_Autoencoder_train import Encoder_P_dataset
from Protein_Autoencoder_train.Encoder_P_Net import AutoEncoder
from model_files import save_load_model
from Molecular_fingureprint_generation import ALL_rawstr_to_fingerprint


device = torch.device('cuda')
train_P_loader = DataLoader(Encoder_P_dataset_evaluate_COVID_19.train_P_set, batch_size = 1, shuffle = False)

model = save_load_model.load('protein_encoder_final')

model = model.to(device)

i=0
dataset_train = []
for item in train_P_loader:
    batch_input = item
    batch_input = batch_input.to(device)
    enc_output = model.encoder(batch_input)
    index = enc_output.cpu().data.numpy()
    index_arr = index.flatten()
    index_arr = index_arr.reshape(64)
    dataset_train.append(index_arr)
    i=i+1
    
print("Number of train data : ",i)

df_train = pd.DataFrame (dataset_train)

df_train.to_csv('train_Protein.csv')
print("train_protein.csv is successfully created")

df = pd.read_csv (r'train_Protein.csv', sep=',', encoding='utf-8') ### change the file name accordingly
df.to_json (r'Protein_AE_output_train.json', lines=False, orient = 'values')
