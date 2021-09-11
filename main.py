"""
@Author: RyanHuang
@Data  : 20210911
"""
import torch
import torch.nn.functional as F
import preprocess as pp
from dataset import TianChiData
from model import TianChiModel

# --------- some 超参数 ---------
train_valid = 0.2 # Valid 占 Train 中的比例, 0.2 的意思是 4 : 1
learning_rate = 0.02

loss_ce_w = 5
loss_mse_w = 1

EPOCH = 20
MODEL_SAVE_PATH = "./output/best.pth"

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
                                          batch_size=2)

validloader = torch.utils.data.DataLoader(valid_Dateset,
                                          shuffle=False,
                                          batch_size=1)

testloader  = torch.utils.data.DataLoader(test_Dataset,
                                          shuffle=False,
                                          batch_size=1)


# --------- 模型初始化 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', input_channel=35, class_out=4, regr_out=3).to(device)
if False:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_func_CE = torch.nn.CrossEntropyLoss(reduction='mean')
loss_func_MSE = torch.nn.MSELoss()

# --------- 模型初始化 ---------

for epc in range(EPOCH):
    for i, (X, y) in enumerate(trainloader):

        # 淦, 最新版本的 DataLoader 加载进来直接就是 Tensor
        # X = torch.from_numpy(X).to(device)
        # y = torch.from_numpy(y).to(device)
        X = X.to(device)
        y = y.to(device)

        # print(X.shape, type(X), X.requires_grad)
        # print(y.shape)

        cls_output, reg_output = model(X)

        loss_mse = loss_func_MSE(reg_output, y[:, :3])
        loss_ce = F.binary_cross_entropy_with_logits(cls_output, y[:, 3:]) # .type(torch.long)

        loss = loss_ce_w * loss_ce + loss_mse_w * loss_mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not i % 5:
            print(loss)
        
        break
    break


torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("保存完毕!!")