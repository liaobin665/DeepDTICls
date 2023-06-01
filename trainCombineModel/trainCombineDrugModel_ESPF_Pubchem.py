"""
author: liaobin
"""
import numpy as np
import tensorflow as tf
import pandas as pd

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
from model.OneDrugModel import MyOneDrugModel as oneDrugCodeModel


drug_ESPF_and_Pubchem_df = pd.read_csv('../data/total_protein_drug_ESPF_and_Pubchem_Lable.csv')
drug_ESPF_and_Pubchem_df =drug_ESPF_and_Pubchem_df.sample(frac=1)

total = drug_ESPF_and_Pubchem_df.to_numpy()

split_position = 40536
# train data
train_protein  = total[0:split_position, 0:2500].astype('float')
train_drug_ESPF_and_Pubchem = total[0:split_position, 2500:5967].astype('float')
train_label = total[0:split_position, 5967:5968].astype('float')

# test data
test_protein = total[split_position:44841, 0:2500].astype('float')
test_drug_ESPF_and_Pubchem  = total[split_position:44841, 2500:5967].astype('float')
test_label = total[split_position:44841, 5967:5968].astype('float')

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_ESPF_and_Pubchem, "label": test_label}
test_dic = {"drug_ESPF_and_Pubchem": test_dic1}

model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 128,
    "dropout": 0
}

drug_ESPF_and_Pubchem_one_model = oneDrugCodeModel(**model_params, drug_len1=train_drug_ESPF_and_Pubchem.shape[1])
drug_ESPF_and_Pubchem_one_model.summary()
drug_ESPF_and_Pubchem_one_model.validation(train_drug_ESPF_and_Pubchem,train_protein,train_label,test_drug_ESPF_and_Pubchem,test_protein,test_label,n_epoch=20,**test_dic)

# writer = tf.compat.v1.summary.FileWriter("./logs", tf.compat.v1.get_default_graph())
# writer.close()
print("stop")