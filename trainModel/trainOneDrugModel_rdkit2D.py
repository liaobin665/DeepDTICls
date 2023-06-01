"""
author: liaobin
备注：rdkit2D 一共是200维，不需要进行CNN，适合用 MyOneDrugModel_Without_Encoding模型
"""
import numpy as np
import tensorflow as tf
import pandas as pd

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
from model.OneDrugModel_without_encoding import MyOneDrugModel_Without_Encoding as oneDrugCodeModel


rdkit2D_df = pd.read_csv('../data/total_protein_drug_rdkit_2d_Lable.csv')
rdkit2D_df =rdkit2D_df.sample(frac=1)

total = rdkit2D_df.to_numpy()

split_position = 40536
# train data
train_protein  = total[0:split_position, 0:2500].astype('float')
train_drug_rdkit2D = total[0:split_position, 2500:2700].astype('float')
train_label = total[0:split_position, 2700:2701].astype('float')

# test data
test_protein = total[split_position:44841, 0:2500].astype('float')
test_drug_rdkit2D = total[split_position:44841, 2500:2700].astype('float')
test_label = total[split_position:44841, 2700:2701].astype('float')

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_rdkit2D, "label": test_label}
test_dic = {"rdkit2D": test_dic1}

model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 128,
    "dropout": 0
}

rdkit2D_one_model = oneDrugCodeModel(**model_params, drug_len1=train_drug_rdkit2D.shape[1])
rdkit2D_one_model.summary()
rdkit2D_one_model.validation(train_drug_rdkit2D,train_protein,train_label, test_drug_rdkit2D,test_protein,test_label,n_epoch=20,**test_dic)

writer = tf.compat.v1.summary.FileWriter("./logs", tf.compat.v1.get_default_graph())
writer.close()
print("stop")