"""
author: liaobin
"""
import numpy as np
import tensorflow as tf
import pandas as pd

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
from model.OneDrugModel_without_encoding import MyOneDrugModel_Without_Encoding as denseModel
from model.OneDrugModel import MyOneDrugModel as sparseModel
# from model.CombinedDrugModel import MyTowDrugModel as twoModel


three_df = pd.read_csv('../../data/total_protein_drug_ESPF_rdkit2D_pubchem_Lable.csv')
three_df =three_df.sample(frac=1)

total = three_df.to_numpy()

split_position = 40357
# train data
train_protein  = total[0:split_position, 0:2500].astype('float')
train_drug_ESPF_rdkit2d_Pubchem = total[0:split_position, 2500:6167].astype('float')
train_label = total[0:split_position, 6167:6168].astype('float')

# test data
test_protein = total[split_position:44841, 0:2500].astype('float')
test_drug_ESPF_rdkit2d_Pubchem  = total[split_position:44841, 2500:6167].astype('float')
test_label = total[split_position:44841, 6167:6168].astype('float')

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_ESPF_rdkit2d_Pubchem, "label": test_label}
test_dic_dense = {"dense_drug_ESPF_Pubchem_rdkit2d_v2": test_dic1}
test_dic_sparse = {"sparse_drug_ESPF_Pubchem_rdkit2d_v2": test_dic1}

model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 64,
    "dropout": 0
}

dense_model = denseModel(**model_params, drug_len1=train_drug_ESPF_rdkit2d_Pubchem.shape[1])
dense_model.summary()
dense_model.validation(train_drug_ESPF_rdkit2d_Pubchem, train_protein, train_label, test_drug_ESPF_rdkit2d_Pubchem, test_protein, test_label, n_epoch=60, **test_dic_dense)

sparse_model = sparseModel(**model_params, drug_len1=train_drug_ESPF_rdkit2d_Pubchem.shape[1])
sparse_model.summary()
sparse_model.validation(train_drug_ESPF_rdkit2d_Pubchem, train_protein, train_label, test_drug_ESPF_rdkit2d_Pubchem, test_protein, test_label, n_epoch=60, **test_dic_sparse)


print("stop")