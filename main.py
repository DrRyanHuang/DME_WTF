"""
@Author: RyanHuang
@Data  : 20210911
"""
import torch
import preprocess as pp
from dataset import TianChiData
from model import TianChiModel

# --------- some 超参数 ---------
train_valid = 0.2 # Valid 占 Train 中的比例, 0.2 的意思是 4 : 1
learning_rate = 0.02


# --------- 数据集划分 ---------
train_num = int(len(pp.new_DF_preprocessed_without_NaN_tr)*(1-train_valid))
valid_num = len(pp.new_DF_preprocessed_without_NaN_tr) - train_num
train_valid_data_df = pp.new_DF_preprocessed_without_NaN_tr.sample(frac=1, random_state=527).reset_index(drop=True)



# --------- 定义读取数据集的对象 ---------
train_Dateset = TianChiData(train_valid_data_df.iloc[:train_num], 
                            "data/OneDrive_2_2021-9-8", mode="train")
valid_Dateset = TianChiData(train_valid_data_df.iloc[train_num:], 
                            "data/OneDrive_2_2021-9-8", mode="valid")
test_Dataset = TianChiData(pp.new_DF_preprocessed_without_NaN_te, 
                           "data/OneDrive_1_2021-9-11", mode="test")


trainloader = torch.utils.data.DataLoader(train_Dateset,
                                          shuffle=True,
                                          batch_size=16)

validloader = torch.utils.data.DataLoader(valid_Dateset,
                                          shuffle=False,
                                          batch_size=16)

testloader  = torch.utils.data.DataLoader(test_Dataset,
                                          shuffle=False,
                                          batch_size=16)


# --------- 模型初始化 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', input_channel=35, num_classes=7).to(device)
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_func_CE = torch.nn.CrossEntropyLoss()
loss_func_MSE = torch.nn.MSELoss()

# --------- 模型初始化 ---------


for i, (X, y) in enumerate(trainloader):

    # 淦, 最新版本的 DataLoader 加载进来直接就是 Tensor
    # X = torch.from_numpy(X).to(device)
    # y = torch.from_numpy(y).to(device)
    X = X.to(device)
    y = y.to(device)

    # print(X.shape, type(X), X.requires_grad)
    # print(y.shape)

    output = model(X)
    print(output)
    print()
    break

