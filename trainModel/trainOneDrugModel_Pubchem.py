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


pubchem_df = pd.read_csv('../data/total_protein_drug_Pubchem_Lable.csv')
pubchem_df =pubchem_df.sample(frac=1)

total = pubchem_df.to_numpy()

split_position = 40536
# train data
train_protein  = total[0:split_position, 0:2500].astype('float')
train_drug_pubchem = total[0:split_position, 2500:3381].astype('float')
train_label = total[0:split_position, 3381:3382].astype('float')

# test data
test_protein = total[split_position:44841, 0:2500].astype('float')
test_drug_pubchem  = total[split_position:44841, 2500:3381].astype('float')
test_label = total[split_position:44841, 3381:3382].astype('float')

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_pubchem, "label": test_label}
test_dic = {"pubchem": test_dic1}

model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 128,
    "dropout": 0
}

pubchem_one_model = oneDrugCodeModel(**model_params, drug_len1=train_drug_pubchem.shape[1])
pubchem_one_model.summary()
pubchem_one_model.validation(train_drug_pubchem,train_protein,train_label,test_drug_pubchem,test_protein,test_label,n_epoch=20,**test_dic)

writer = tf.compat.v1.summary.FileWriter("./logs", tf.compat.v1.get_default_graph())
writer.close()
print("stop")