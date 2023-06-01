"""
author: liaobin
功能：加载预训练模型，加载 训练/测试数据  获取中间层输出结果。
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


# 加载模型，并得到子模型
load_model = tf.keras.models.load_model('compareClassfier_model.h5')
sub_model = tf.keras.models.Model(inputs=load_model.input, outputs = load_model.get_layer('batch_normalization_6').output)

print(sub_model.summary())

# 输入数据进行预测
test_middle_X = sub_model.predict([test_drug_pubchem,test_protein])
train_middle_X = sub_model.predict([train_drug_pubchem,train_protein])

# 改变数据形状，并保存。
test_middle_X = test_middle_X.reshape(-1,256)
train_middle_X = train_middle_X.reshape(-1,256)

pd.DataFrame(test_middle_X).to_csv('test_middle_X.csv')
pd.DataFrame(train_middle_X).to_csv('train_middle_X.csv')
