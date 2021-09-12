"""
@Author: RyanHuang
@Data  : 20210911
"""
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
import preprocess as pp
from dataset import TianChiData
from model import TianChiModel

MODEL_SAVE_PATH = "./output/best.pth"

# --------- 定义读取数据集的对象 ---------
test_Dataset = TianChiData(pp.new_DF_preprocessed_without_NaN_te, 
                           "data/OneDrive_1_2021-9-11", mode="test")


testloader  = torch.utils.data.DataLoader(test_Dataset,
                                          shuffle=False,
                                          batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', input_channel=35, class_out=4, regr_out=3).to(device)
if True:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))


with torch.no_grad(): 
    out_array_list = []
    names_array = []

    for (X, names) in tqdm(testloader):
        X = X.to(device)
        
        cls_out, reg_out = model(X)

        cls_out = cls_out.cpu().numpy()
        reg_out = reg_out.cpu().numpy()

        out_array = np.concatenate([reg_out, cls_out], axis=-1)

        out_array_list.append(out_array)
        names_array += names


    out_array = np.concatenate(out_array_list)

    out_pd = pd.DataFrame(out_array, columns=['continue injection', 'IRF', 'SRF', 'HRF', 'preCST', 'VA', 'CST'])
    out_pd['patient ID'] = names_array

    def guiyi_inves(key):
        
        return (out_pd[key] + 1) / 2 + \
            (pp.some_interesting_dict["max_"+key] - pp.some_interesting_dict["min_"+key]) +\
            pp.some_interesting_dict["min_"+key]

    out_pd['preCST'] = guiyi_inves("preCST")
    out_pd["CST"] = guiyi_inves("CST")
    out_pd["VA"] = guiyi_inves("VA")

    # out_pd["preCST"] = pp.train_data_df["preCST"].mean()
    # out_pd["VA"] = pp.train_data_df["VA"].mean()
    # out_pd["CST"] = pp.train_data_df["CST"].mean()

    # out_pd["continue injection"] = 0
    # out_pd["IRF"] = 0
    # out_pd["SRF"] = 0
    # out_pd["HRF"] = 0
    
    out_pd.to_csv("./data/out.csv", index=False)