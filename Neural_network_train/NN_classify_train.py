import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sys
sys.path.append('..')

from Neural_network_train.NN_data_set import NN_data_set
from Neural_network_train.NN_classify_Net import NN
from Save_models import save_load_model

device = torch.device('cuda')

print("Loading Neural network train data started")
train_set = NN_data_set(train = True)
train_loader = DataLoader(train_set,batch_size = 1024,shuffle = True)


model = NN()
model.train()
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

print("Loading Neural network test data started")
test_set = NN_data_set(train = False)
test_loader = DataLoader(test_set,batch_size = 1)

def Get_ACC():
    cnt = 0
    correct = 0
    model.eval()
    loss_sum = 0
    for item in test_loader:
        batch_inputs, batch_labels = item
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(batch_inputs)
        loss_sum+=criterion(out,batch_labels)
        yu_zhi = 0.5
        cur_label = batch_labels[0].data.item()
        cur_out = out[0].data.item()
        if ((cur_out > yu_zhi and cur_label == 1) or (cur_out < yu_zhi and cur_label == 0)):
            correct += 1
        cnt += 1
    model.train()
    acc = correct / cnt *100
    eval_loss = loss_sum/cnt
    return acc,eval_loss



best_model = {'acc':0,'loss':0}

avg_loss_all = []
avg_accuracy_all = []
print("Neural network training started...")

for epoch in range(5000):
    cnt = 0
    sum_loss = 0
    for item in train_loader:
        batch_inputs,batch_labels = item
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        out = model(batch_inputs)
        loss = criterion(out,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.item()
        cnt+=1
    ave_loss = sum_loss/cnt
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

    if((epoch+1)%50==0):
        avg_loss_all.append(ave_loss)
        acc,eval_loss = Get_ACC()
        print('eval: acc:{}, eval loss:{}'.format(acc,eval_loss))
        print('best acc:{}, -> eval loss:{}'.format(best_model['acc'],best_model['loss']))
        avg_accuracy_all.append(acc)
        if(acc>best_model['acc']):
            best_model = {'acc':acc, 'loss':eval_loss}
            print('this is best!')


print('best acc:{}, best eval loss:{}'.format(best_model['acc'],best_model['loss']))


with open('avg_loss_all.json','w') as f:
    json.dump(avg_loss_all,f)
with open('avg_accuracy_all.json','w') as f:
    json.dump(avg_accuracy_all,f)


plt.figure(1)
plt.cla()
plt.plot(avg_loss_all, 'go-', linewidth=2, markersize=4,
         markeredgecolor='red', markerfacecolor='m')
plt.pause(0.000000001)

plt.figure(2)
plt.cla()
plt.plot(avg_accuracy_all, 'go-', linewidth = 2, markersize = 4,
         markeredgecolor = 'red', markerfacecolor = 'm')
plt.pause(0.000000001)

plt.show()

save_load_model.save(model, 'NN_classify_final')