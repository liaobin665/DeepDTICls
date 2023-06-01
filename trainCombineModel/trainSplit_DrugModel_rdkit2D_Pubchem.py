"""
author: liaobin
"""
import numpy as np
import tensorflow as tf
import pandas as pd

# 保证sess.run()能够正常运行
tf.compat.v1.disable_eager_execution()
# import tensorflow.keras.backend as K
from model.TwoDrugModel_with_BiRNN import MyTowDrugModel_with_BiRNN as towDrugModel_with_BiRNN


drug_rdkit2D_and_Pubchem_df = pd.read_csv('../data/total_protein_drug_rdkit2D_and_Pubchem_Lable.csv')
drug_rdkit2D_and_Pubchem_df =drug_rdkit2D_and_Pubchem_df.sample(frac=1)

protein = drug_rdkit2D_and_Pubchem_df.loc[:, '0_pro':'2499_pro']
drug_rdkit2D = drug_rdkit2D_and_Pubchem_df.loc[:, '0_rdkit':'199_rdkit']
drug_pubchem = drug_rdkit2D_and_Pubchem_df.loc[:, '0_pubchem':'880_pubchem']
label = drug_rdkit2D_and_Pubchem_df.loc[:, 'Label']

total = drug_rdkit2D_and_Pubchem_df.to_numpy()

split_position = 40536

train_protein  = protein.to_numpy()[0:split_position,:].astype('float')
test_protein = protein.to_numpy()[split_position:44841,:].astype('float')

train_drug_rdkit2D = drug_rdkit2D.to_numpy()[0:split_position,:].astype('float')
test_drug_rdkit2D = drug_rdkit2D.to_numpy()[split_position:44841,:].astype('float')

train_drug_pubchem = drug_pubchem.to_numpy()[0:split_position,:].astype('float')
test_drug_pubchem = drug_pubchem.to_numpy()[split_position:44841,:].astype('float')

train_label = label.to_numpy()[0:split_position].astype('float')
test_label = label.to_numpy()[split_position:44841].astype('float')

test_dic1 = {"protein_feature": test_protein, "drug_feature1": test_drug_pubchem, "drug_feature2": test_drug_rdkit2D, "label": test_label}
test_dic = {"pubchem_rdkit2D_with_BiRNN": test_dic1}

model_params = {
    "decay": 0.0001,
    "activation": "elu",
    "filters": 128,
    "dropout": 0
}

two_model = towDrugModel_with_BiRNN(**model_params, drug_len1=train_drug_pubchem.shape[1], drug_len2=train_drug_rdkit2D.shape[1])
two_model.summary()
two_model.validation(train_drug_pubchem, train_drug_rdkit2D, train_protein, train_label, test_drug_pubchem, test_drug_rdkit2D, test_protein, test_label, n_epoch=60, **test_dic)

writer = tf.compat.v1.summary.FileWriter("./logs", tf.compat.v1.get_default_graph())
writer.close()
print("stop")