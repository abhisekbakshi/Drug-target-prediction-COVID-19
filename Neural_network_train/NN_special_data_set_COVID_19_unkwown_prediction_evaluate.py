
import json
import torch
from torch.utils.data import Dataset
from torch.autograd import  Variable

# from train_model.train_NN import all_fingerprint_to_code

class NN_special_data_set_COVID_19_unkwown_prediction_evaluate(Dataset):
    def __init__(self,train=True):
        self.all_tensor = []
        List = []
        if(train):
            print("") # the statement has no use
        else:
            f_name = input("Enter the NN unknown binding evaluate file name with extension (.json) : ")
            with open(f_name,'r') as f:
                List = json.load(f)

        for item in List:
            i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47, i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63, i64, i65, i66, i67, i68, i69, i70, i71, i72, i73, i74, i75, i76, i77, i78, i79, i80, i81, i82, i83, i84, i85, i86, i87, i88, i89, i90, i91, i92, i93, i94, i95, i96, i97, i98, i99, i100, i101, i102, i103, i104, i105, i106, i107, i108, i109, i110, i111, i112, i113, i114, i115, i116, i117, i118, i119, i120, i121, i122, i123, i124, i125, i126, i127, i128 = item
            input_code = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20,
                          i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34, i35, i36, i37, i38, i39,
                          i40, i41, i42, i43, i44, i45, i46, i47, i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58,
                          i59, i60, i61, i62, i63, i64, i65, i66, i67, i68, i69, i70, i71, i72, i73, i74, i75, i76, i77,
                          i78, i79, i80, i81, i82, i83, i84, i85, i86, i87, i88, i89, i90, i91, i92, i93, i94, i95, i96,
                          i97, i98, i99, i100, i101, i102, i103, i104, i105, i106, i107, i108, i109, i110, i111, i112,
                          i113, i114, i115, i116, i117, i118, i119, i120, i121, i122, i123, i124, i125, i126, i127,
                          i128]

            input_code = torch.FloatTensor(input_code)
            self.all_tensor.append([input_code])


    def __getitem__(self, index):
        # return Variable(self.all_tensor[index][0]),Variable(self.all_tensor[index][1])
        return Variable(self.all_tensor[index][0])

    def __len__(self):
        return len(self.all_tensor)
