import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from Training_testing_data_design import ALL_rawstr_to_fingerprint


class Drug_Encoder_DataSet(Dataset):
    def __init__(self, train = True):
        all_D_input = []
        if (train):
            print("Drug Autoencoder train dataset creation started")
            for item in ALL_rawstr_to_fingerprint.train_fingerprint:
                P_f, D_f, label = item
                all_D_input.append(D_f)
        else:
            print("Drug Autoencoder test dataset creation started")
            for item in ALL_rawstr_to_fingerprint.test_fingerprint:
                P_f, D_f, label = item
                all_D_input.append(D_f)
        print('list of Drug length:{}'.format(len(all_D_input)))
        self.D_tensor = torch.FloatTensor(all_D_input)

    def __getitem__(self, index):
        return Variable(self.D_tensor[index])

    def __len__(self):
        return len(self.D_tensor)


train_D_set = Drug_Encoder_DataSet(train = True)
test_D_set = Drug_Encoder_DataSet(train = False)
