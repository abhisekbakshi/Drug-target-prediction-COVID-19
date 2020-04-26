import json

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Neural_network_train.NN_special_data_set_COVID_19_unkwown_prediction_evaluate import NN_special_data_set_COVID_19_unkwown_prediction_evaluate

from model_files import save_load_model

device = torch.device('cuda')

test_set_COVID_19 = NN_special_data_set_COVID_19_unkwown_prediction_evaluate(train = False)
test_loader_COVID_19 = DataLoader(test_set_COVID_19, batch_size = 1, shuffle = True)

model = save_load_model.load('NN_classify_final')
model = model.to(device)
model.eval()

i=1
all_out_COVID_19_evaluate = []
for item in test_loader_COVID_19:
    batch_inputs = item
    batch_inputs = Variable(batch_inputs).to(device)

    out = model(batch_inputs)
    cur_out_COVID_19 = out[0].data.item()
    all_out_COVID_19_evaluate.append([cur_out_COVID_19])

    print("\nSet ",i," predicted")
    i=i+1

print(all_out_COVID_19_evaluate)
with open('all_out_COVID_19_evaluate_unknown_binding.json','w') as f:
    json.dump(all_out_COVID_19_evaluate,f)





