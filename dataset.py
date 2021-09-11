"""
@Author: RyanHuang
@Data  : 20210910
"""
import os
import cv2
import numpy as np
from pandas.io.formats.format import return_docstring
import preprocess as pp
import torch
from torch.utils.data import Dataset

# 将数据集 4 : 1 分开
# trainSet、validSet 和 TestSet



class TianChiData(Dataset):

    # 所有照片的后缀 有的图片文件夹 tmd 只有
    # SUFFIX = ["_100{}.jpg".format(i) for i in range(6)] + ["_200{}.jpg".format(i) for i in range(6)]
    SUFFIX = ["_100{}.jpg".format(i) for i in range(6)]

    Y_start, Y_end = 0, 470 / 561
    X_start, X_end = 467/1189, 1

    def __init__(self, data_DF, img_root, mode="Train"):
        '''
        @Param:
            img_root 是包含 `0000-0000` 、 `0000-0004L`、`0000-0005` 等文件夹的目录
        '''
        self.data_DF = data_DF
        self.img_root = img_root
        self.mode = mode

        self.img_dir_list = os.listdir(img_root)

    def __getitem__(self, index):

        currrent_Series = self.data_DF.iloc[index]
        patient_ID = currrent_Series["patient ID"]

        # 当前 patient_ID 对应图片路径的 path
        img12_list = self.__match_dir(patient_ID)

        # 有些图片即使是后处理加东西也不行, 这里需要改一改, 手动指定路径, FUCK!!
        if os.path.basename(img12_list[0]).startswith("0000-1765R"):
            print("WARNING", "0000-1765R", "FUCK!!")
            img12_list = [
                os.path.join(self.img_root, "0000-1765", "0000-1765R_2 ({}).jpg".format(i)) for i in range(1, 7)
            ]

        # 读取所有图片并堆叠
        img_concat = self.__read_img(img12_list)
        number_info = currrent_Series[:17].values    # <------- 注意此处是改特征的地方

        # 这一步做特征融合, 也不知道效果好不好
        number_info_array = np.tile(number_info[:, None, None], 
                                    (1, img_concat.shape[1], img_concat.shape[2]))
        number_info_array = np.concatenate([img_concat, number_info_array], axis=0) 

        if self.mode != 'test':
            label_info  = currrent_Series[17:-1].values  # -1 是删去病人 ID 的意思               
            return number_info_array.astype(np.float32), label_info.astype(np.float32)
        else:
            return number_info_array.astype(np.float32)

    def __len__(self):
        return len(self.data_DF)

    def __match_dir(self, patient_ID):
        """
        根据 `0000-0004L` 匹配 12 张照片路径
        """
        patient_ID_copy = patient_ID
        if patient_ID not in self.img_dir_list:
            # 要是这个 patient_ID 是 `0000-0004L`, 而 `self.img_dir_list` 中有 `0000-0004` , 所以删去最后一位
            patient_ID = patient_ID[:-1]
        img_12_path = os.path.join(self.img_root, patient_ID, patient_ID_copy)
        img_12 = [img_12_path+suffix for suffix in self.SUFFIX]
        
        return img_12

    def __read_img(self, img_12_list):
        '''
        根据 img_12_list 读取图片
        并作图片裁剪, 并堆叠
        '''
        img_list = []
        for img_path in img_12_list:

            # ---------- Data 问题 `0000-1315R_1000.jpg` 不存在 只有 `0000-1315L_1000.jpg` ----------
            # ---------- 而提交的 submit.csv 要求有 `0000-1315R` 所以将 `0000-1315L_1000.jpg` 用于预测 ----------
            # ---------- 以上两行注释作废, 数据集搞错了 ----------

            if not os.path.exists(img_path):
                print("WARNING: ", os.path.basename(img_path), " NOT EXIST!!")
                # ---------- Train: `0000-0732R_1005` `0000-0327L` `0000-0708L_1005`
                # ----------        `0000-1536R` `0000-1893R` `0000-1102L` `0000-1958R_1003`
                # ----------        `0000-1942L_1005` `0000-0162R_1003` `0000-1903R`
                # ----------        `0000-0877L_1005` `0000-0159L_1005` `0000-0159R`
                # ----------        `0000-0842L_1005` `0000-1700L` `0000-1700R` `0000-1765R_1000`......
                # ---------- Test : `0000-0715R_1000` 不存在  `0000-0982R_1000` `0000-0924R_1005`
                img_path = img_path.replace("100", "200")

                if not os.path.exists(img_path):
                    # 本来 100 -> 200 应该是可以直接过去的, 结果 `0000-0025` 没有 `0000-0025L`
                    print("WARNING: ", os.path.basename(img_path), " NOT EXIST TWICE!!")
                    img_path = self.__change_name(img_path)

            img_current = cv2.imread(img_path)
            # 进行图像裁剪
            H, W, _ = img_current.shape
            img_current = img_current[int(H*self.Y_start):int(H*self.Y_end), 
                                      int(W*self.X_start):int(W*self.X_end)]

            # print(int(H*self.Y_start), int(H*self.Y_end), int(W*self.X_start), int(W*self.X_end))

            # cv2.imshow(img_path, img_current)
            # cv2.imshow("fuck124", cv2.imread(img_path))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img_list.append(img_current)

        img_array = np.concatenate(img_list, axis=-1)
        return img_array.transpose(2, 0, 1)

    def __change_name(self, mistake_file_path):
        # 一个几乎用不到的函数...
        wrong_file_path, wrong_file_name = os.path.split(mistake_file_path)
        all_data_dir, wrong_file_parent = os.path.split(wrong_file_path)

        if "R" in wrong_file_name:
            wrong_file_name = wrong_file_name.replace("R", "L")
            # wrong_file_parent += "L"
        else:
            wrong_file_name = wrong_file_name.replace("L", "R")
            # wrong_file_parent += "R"

        return os.path.join(all_data_dir, wrong_file_parent, wrong_file_name)




# import preprocess as pp

# train_Dateset = TianChiData(pp.new_DF_preprocessed_without_NaN_tr, 
#                             "data/OneDrive_2_2021-9-8", mode="train")
# # Valid_Dateset = 

# test_Dataset = TianChiData(pp.new_DF_preprocessed_without_NaN_te, 
#                            "data/OneDrive_1_2021-9-11", mode="test")
