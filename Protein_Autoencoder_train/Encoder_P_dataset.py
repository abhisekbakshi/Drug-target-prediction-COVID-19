import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from Training_testing_data_design import ALL_rawstr_to_fingerprint


class Protein_Encoder_DataSet(Dataset):
    def __init__(self, train = True):
        all_P_input = []
        if (train):
            print("Protein Autoencoder train dataset creation started")
            for item in ALL_rawstr_to_fingerprint.train_fingerprint:
                P_f, D_f, label = item
                all_P_input.append(P_f)
        else:
            print("Protein Autoencoder test dataset creation started")
            for item in ALL_rawstr_to_fingerprint.test_fingerprint:
                P_f, D_f, label = item
                all_P_input.append(P_f)
        print('list Protein len:{}'.format(len(all_P_input)))
        self.P_tensor = torch.FloatTensor(all_P_input)

    def __getitem__(self, index):
        return Variable(self.P_tensor[index])

    def __len__(self):
        return len(self.P_tensor)


train_P_set = Protein_Encoder_DataSet(train = True)
test_P_set = Protein_Encoder_DataSet(train = False)

