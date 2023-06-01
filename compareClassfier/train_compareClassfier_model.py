"""
author: liaobin
"""
import numpy as np
import tensorflow as tf
import pandas as pd

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
from model.OneDrugModel import MyOneDrugModel as oneDrugModel


# fixed train data
train_protein = pd.read_csv('../dataPubChem/pubchem_train_protein.csv').to_numpy()
train_drug_pubchem = pd.read_csv('../dataPubChem/pubchem_train_drug.csv').to_numpy()
train_label = pd.read_csv('../dataPubChem/pubchem_y_train.csv').to_numpy()


# fixed test data
test_protein = pd.read_csv('../dataPubChem/pubchem_test_protein.csv').to_numpy()
test_drug_pubchem = pd.read_csv('../dataPubChem/pubchem_test_drug.csv').to_numpy()
test_label = pd.read_csv('../dataPubChem/pubchem_y_test.csv').to_numpy()

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_pubchem, "label": test_label}
test_dic = {"cnn_model_fliters_128": test_dic1}


model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 128,
    "dropout": 0
}


model = oneDrugModel(**model_params, drug_len1=train_drug_pubchem.shape[1])
model.summary()
model.validation(train_drug_pubchem,train_protein,train_label,test_drug_pubchem,test_protein,test_label,n_epoch=60,**test_dic)
model.save("compareClassfier_model.h5")

writer = tf.compat.v1.summary.FileWriter("./logs", tf.compat.v1.get_default_graph())
writer.close()
print("stop")