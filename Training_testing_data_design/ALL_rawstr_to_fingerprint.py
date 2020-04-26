import json
import os

import pandas as pd
import project_path
from Molecular_fingureprint_generation import get_fingerprint_drug
from Molecular_fingureprint_generation import get_fingerprint_protein


def transfer(dataset):
    '''return list of list [[],[],[]] of feature
    :rtype: list
    '''
    data_feature = []
    print('processing {} to fingerprints'.format(dataset))
    with open(dataset, 'r') as f:
        data_raw = json.load(f)
        i=1
        j=0
        for item in data_raw:
            P_str, D_str, label = item
            P_f = get_fingerprint_protein.get_fingerprint_from_protein_squeeze(P_str)
            try:
                D_f = get_fingerprint_drug.get_fingerprint_from_smiles(D_str)
                tmp1 = [P_f, D_f, label]
                data_feature.append(tmp1)
            except:
                j=j+1
                print(i," number item passed\n")
                pass
            i = i+1
    print("\ntotal item skipped = ",j)
    print('fingerprints process OK!')
    return data_feature


f_name_train = 'train_data_all_shuffled.json' ## change the json file name according to Autoencoder train/extract autoencoder output
f_name_test = 'test_set_2.json'

train_fingerprint = transfer(os.path.join(project_path.PROJECT_ROOT, f_name_train))
test_fingerprint = transfer(os.path.join(project_path.PROJECT_ROOT, f_name_test))

df_train = pd.DataFrame (train_fingerprint)
df_test = pd.DataFrame (test_fingerprint)


f_train = input("Enter csv file name with exension to store train fingurprint :")
f_test = input("Enter csv file name with exension to store test fingurprint :")

df_train.to_csv(f_train)
print(f_train," is successfully created")
df_test.to_csv(f_test)
print(f_test," is successfully created")


'''train_data_feature input to encoder: ->torch.FloatTensor() -> Variable().to(device)'''
