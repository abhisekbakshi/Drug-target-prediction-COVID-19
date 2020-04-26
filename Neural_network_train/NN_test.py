import json

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Neural_network_train.NN_data_set import NN_data_set

from model_files import save_load_model

device = torch.device('cuda')

test_set = NN_data_set(train = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)


model = save_load_model.load('NN_classify_final')
model = model.to(device)
model.eval()
print("Neural network classification model loaded")


sum = 0
correct = 0
NN_test_out_all = []
for item in test_loader:
    batch_inputs, batch_labels = item
    batch_inputs = Variable(batch_inputs).to(device)
    batch_labels = Variable(batch_labels).to(device)

    out = model(batch_inputs)
    yuzhi = 0.5
    cur_label = batch_labels[0].data.item()
    cur_out = out[0].data.item()
    NN_test_out_all.append([cur_out,cur_label])
    if ((cur_out > yuzhi and cur_label == 1) or (cur_out < yuzhi and cur_label == 0)):
         correct += 1
    print(cur_out,cur_label)
    sum += 1

print('acc:{}'.format(correct / sum * 100))

f_name = input("Enter file name with extension to store NN test outputs with predicted level: ")
with open(f_name,'w') as f:
    json.dump(NN_test_out_all,f)




