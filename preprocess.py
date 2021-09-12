"""
@Author: RyanHuang
@Data  : 20210908
"""

# patient ID           | 
# gender               |  + one-hot         |  无 NaN
# age                  |  + Normalization   |  无 NaN
# diagnosis            |  + one-hot         |  无 NaN
# preVA                |  + Normalization   |  有 NaN
# anti-VEGF            |  + one-hot         |  无 NaN   | 备注 12369
# preCST               |  连续值*           |  
# preIRF               |  二分类     -
# preSRF               |  二分类     -
# prePED               |  二分类     -
# preHRF               |  二分类     -
# VA                   |  连续值*           |
# continue injection   |  二分类*           |
# CST                  |  连续值*           |
# IRF                  |  二分类*           |
# SRF                  |  二分类*           |
# PED                  |  二分类     -
# HRF                  |  二分类*           |

# + 是 Train 和 Test 都有的
# * 是要预测的
# - 是用不着的

import pandas as pd
import numpy as np
import math

train_data_df = pd.read_csv('./data/TrainingAnnotation.csv')
test_data_df = pd.read_csv('./data/PreliminaryValidationSet_Info.csv')

# print(len(train_data_df))
# print(len(test_data_df))

# 第一次训练的时候 只要有 NaN, 则就删除
train_data_df_without_nan = train_data_df.dropna(how="any")
test_data_df_without_nan  = test_data_df.dropna(how="any")

# print(len(train_data_df_without_nan))
# print(len(test_data_df_without_nan))

def haveNaN(train_data=None, test_data=None):
    """
    @Brife:
        判断某个 column 是否有 nan
    @Notice:
        建议 train_data 和 test_data 都是 `df.xx.values`
    """
    if test_data is None:
        data_sum = train_data.sum()
    else:
        data_sum = np.concatenate([train_data, test_data]).sum()

    return math.isnan(data_sum)



def deal_age(mean_std_tuple=None, 
             column_train=train_data_df.age.values, 
             column_test=test_data_df.age.values):
    '''
    @Param:

    @Brife:
        建议归一化之后标准化，此处仅进行标准化
    '''
    column_all = np.concatenate([column_train, column_test])

    if mean_std_tuple is None:
        mean = np.mean(column_all)
        std = np.std(column_all)
        mean_std_tuple = (mean, std)
    else:
        mean, std = mean_std_tuple

    column_train_new = (column_train - mean) / std
    column_test_new = (column_test - mean) / std

    return mean_std_tuple, column_train_new, column_test_new



def deal_gender(column=train_data_df.gender.values):
    '''
    @Brife:
        将性别转化为 one_hot

    @Notice:
        原来是     1:男 2:女
        整体减1    0:男 1:女

        MD 一共就两类, 2维 one-hot 个毛线啊
    '''
    column = column - 1
    # return np.eye(2)[column]
    return column



def deal_diagnosis(column=train_data_df.diagnosis.values, map_dict=None):
    '''
    @Brife:
        将 diagnosis 转化为 one_hot
    '''
    if map_dict is None:
        map_dict = {v:k for (k, v) in enumerate(set(column))}
    column_new = np.array([map_dict[x] for x in column])

    return map_dict, np.eye(len(map_dict))[column_new]



def deal_preVA(mean_std_tuple=None, 
               column_train=train_data_df.preVA.values, 
               column_test=test_data_df.age.values):
    '''
    @Param:

    @Brife:
        建议归一化之后标准化，此处仅进行标准化
    @Notice:
        操作一样, 直接调用之前的就行
    '''

    return deal_age(mean_std_tuple=mean_std_tuple, 
                    column_train=column_train, 
                    column_test=column_test)
    
    # column_all = np.concatenate([column_train, column_test])

    # if mean_std_tuple is None:
    #     mean = np.mean(column_all)
    #     std = np.std(column_all)
    #     mean_std_tuple = (mean, std)
    # else:
    #     mean, std = mean_std_tuple

    # column_train_new = (column_train - mean) / std
    # column_test_new = (column_test - mean) / std

    # return mean_std_tuple, column_train_new, column_test_new



def __del_NaN_mean(column):
    """
    @Brife:
        删掉一个 column 中的 NaN 之后求均值
    @Return:
        num_NaN 该 column 中的 NaN 数字
        mean_column_without_NaN 表示除去 NaN 的均值
    """
    
    def NaN2Zero(someone):
        if math.isnan(someone):
            return 0
        return someone

    # 数一下 column 中的 NaN 的数量, 和下边那四行的 for 循环一样
    num_NaN = len(column) - column.count() 
    # num_NaN = 0
    # for x in column:
    #     if math.isnan(x):
    #         num_NaN += 1

    column_new = column.apply(NaN2Zero)
    mean_column_without_NaN = column_new.sum() / column.count() 
    return num_NaN, mean_column_without_NaN



def fill_preVA_NaN(train_column, test_column=None, apply_func=None):
    '''
    @Param:
        train_column  pandas 的 Series
        test_column   pandas 的 Series
    @Brife:
        
    '''
    if test_column is None:
        column = train_column
    else:
        column = pd.concat([train_column, test_column])

    _, mean = __del_NaN_mean(column)
    if apply_func is None:
        apply_func = lambda x: (mean if math.isnan(x) else x)

    train_column_new = train_column.apply(apply_func)

    if test_column is None:
        return train_column_new

    test_column_new = test_column.apply(apply_func)
    return train_column_new, test_column_new



def deal_anti_VEGF(column=train_data_df_without_nan['anti-VEGF'].values, map_dict=None):
    
    return deal_diagnosis(column=column, map_dict=map_dict)



def deal_preCST(column, min_=None, max_=None):
    """
    @Brife:
        MD 刚发现是连续值, 用 min max 映射到 [-1, 1] 吧
    @Param:
        建议 column 是 np.ndarray
    """
    map_small = -1
    map_big   = 1

    if min_ is None:
        min_ = np.min(column)
    if max_ is None:
        max_ = np.max(column)
    column_new = (column - min_) / (max_ - min_) * (map_big - map_small) + map_small
    return min_, max_, column_new



def deal_VA(column, min_=None, max_=None):
    """
    @Brife:
        MD 刚发现是连续值, 用 min max 映射到 [-1, 1] 吧
    """
    return deal_preCST(column, min_, max_)



def deal_CST(column, min_=None, max_=None):
    """
    @Brife:
        MD 刚发现是连续值, 用 min max 映射到 [-1, 1] 吧
    """
    return deal_preCST(column, min_, max_)



# +++++++++++++++++ 原注释部分会有 warning (暂未解决) ++++++++++++++++++
# ----------------- 预处理性别部分 -----------------
# train_data_df_without_nan.loc[:, ("gender")] = train_data_df_without_nan.gender.values - 1
new_gender_tr = train_data_df_without_nan.gender.values - 1
# test_data_df_without_nan.loc[:, ("gender")] = test_data_df_without_nan.gender.values  - 1
new_gender_te = test_data_df_without_nan.gender.values  - 1


# ----------------- 预处理年龄部分 -----------------
mean_std_tuple_age, _, _ = deal_age()      # 注意这里的均值是怎么算的, 用的没缺少 NaN 的值
_, new_age_tr, new_age_te = deal_age(mean_std_tuple_age, 
                                     train_data_df_without_nan.age.values,
                                     test_data_df_without_nan.age.values)
# train_data_df_without_nan.age = column_age_train
# test_data_df_without_nan.age = column_age_test

# ----------------- 预处理 diagnosis -----------------
map_dict, new_diagnosis_tr = deal_diagnosis(train_data_df_without_nan.diagnosis.values)
_, new_diagnosis_te = deal_diagnosis(test_data_df_without_nan.diagnosis.values, map_dict)

# ------------------- 预处理 preVA -------------------
mean_std_tuple_preVA_, new_preVA_tr, new_preVA_te = deal_preVA(None,       # 注意这里的均值是怎么算的, 用的缺少 NaN 的值
                                                               train_data_df_without_nan.preVA.values, 
                                                               test_data_df_without_nan.preVA.values)


# ------------------- 预处理 anti_VEGF -------------------
map_dict, new_anti_VEGF_tr = deal_anti_VEGF(train_data_df_without_nan["anti-VEGF"].values)
_, new_anti_VEGF_te = deal_anti_VEGF(test_data_df_without_nan["anti-VEGF"].values, map_dict)


# ------------------- 预处理 preCST(minmax归一化) -------------------
min_preCST, max_preCST, new_preCST = deal_preCST(train_data_df_without_nan.preCST.values, min_=None, max_=None)


# ------------------- 预处理 VA(minmax归一化) -------------------
min_VA, max_VA, new_VA = deal_VA(train_data_df_without_nan.VA.values, min_=None, max_=None)


# ------------------- 预处理 CST(minmax归一化) -------------------
min_CST, max_CST, new_CST = deal_CST(train_data_df_without_nan.CST.values, min_=None, max_=None)


# ------------------- 新数据合并 -------------------
new_add_array_tr = np.concatenate(
    [
    new_gender_tr.reshape(-1, 1), 
    new_age_tr.reshape(-1, 1),
    new_diagnosis_tr, 
    new_preVA_tr.reshape(-1, 1),
    new_anti_VEGF_tr,
    new_preCST.reshape(-1, 1),
    new_VA.reshape(-1, 1),
    new_CST.reshape(-1, 1)],
    axis=1
)

# 在 Train 数据中要加上 ID 号和标签
lable_tr = train_data_df_without_nan[
    # ['preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF', "patient ID"]
    ['continue injection', 'IRF', 'SRF', 'HRF', "patient ID"]
].values

new_add_array_tr = np.concatenate(
    [new_add_array_tr, lable_tr],
    axis=1
)

new_add_array_te = np.concatenate(
    [
    new_gender_te.reshape(-1, 1), 
    new_age_te.reshape(-1, 1), 
    new_diagnosis_te, 
    new_preVA_te.reshape(-1, 1), 
    new_anti_VEGF_te],
    axis=1
)

new_add_array_te = np.concatenate(
    [new_add_array_te, test_data_df_without_nan["patient ID"].values.reshape(-1, 1)],
    axis=1
)

# ------------------- 新数据列名 -------------------
new_add_column_name = \
    ["new_gender"] + \
    ["norm_age"] + \
    ["new_diagnosis_{}".format(i) for i in range(7)] + \
    ["new_preVA"] + \
    ["new_anti_VEGF_{}".format(i) for i in range(7)]

new_add_column_name_tr = new_add_column_name + \
    ['preCST', 'VA', 'CST', 'continue injection', 'IRF', 'SRF', 'HRF', "patient ID"]  # CST 和 continue... 换了位置
    # ['preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF', "patient ID"] 
    

new_add_column_name_te = new_add_column_name + \
    ["patient ID"]

# print(new_add_array_tr.shape, new_add_array_te.shape, len(new_add_column_name))

# ------------------- 所有新数据合并 -------------------
new_DF_preprocessed_without_NaN_tr = pd.DataFrame(new_add_array_tr, columns=new_add_column_name_tr)

# 这样合并可能会因为ID问题产生空值, 目前该BUG原因未知
# new_DF_preprocessed_without_NaN_tr[
#     ['preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF', "patient ID"]
# train_data_df_without_nan[
#     ['preCST', 'VA', 'continue injection', 'CST', 'IRF', 'SRF', 'HRF', "patient ID"]
# ].values

new_DF_preprocessed_without_NaN_te = pd.DataFrame(new_add_array_te, columns=new_add_column_name_te)
new_DF_preprocessed_without_NaN_te["patient ID"] = test_data_df_without_nan["patient ID"]

new_DF_preprocessed_without_NaN_tr.to_csv("./data/new_DF_preprocessed_without_NaN_tr.csv", index=False)
new_DF_preprocessed_without_NaN_te.to_csv("./data/new_DF_preprocessed_without_NaN_te.csv", index=False)


some_interesting_dict = dict(
    mean_std_tuple_age = mean_std_tuple_age,
    mean_std_tuple_preVA_=mean_std_tuple_preVA_,
    min_preCST=min_preCST, 
    max_preCST=max_preCST,
    min_VA=min_VA, 
    max_VA=min_VA,
    min_CST=min_CST, 
    max_CST=max_CST
)