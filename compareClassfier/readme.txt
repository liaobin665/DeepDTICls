比较最后分类器的效果。
步骤：
第一步：切分数据。得到训练 测试数据集：
第二步：用训练数据集训练模型。
第三步：加载模型，并将测试数据输入 第二步中得到的模型。
第四步：获取中间层的输出，作为预测分类器的输入X，与测试数据集的y，组成新的数据集。
第五步：用第四步中得到的数据集对比分析各分类模型的性能。并于本文模型进行对比。

训练数据是：
# fixed train data
train_protein = pd.read_csv('../dataPubChem/pubchem_train_protein.csv').to_numpy()
train_drug_pubchem = pd.read_csv('../dataPubChem/pubchem_train_drug.csv').to_numpy()
train_label = pd.read_csv('../dataPubChem/pubchem_y_train.csv').to_numpy()

测试数据是：（或者说，模型的测试输入是）
# fixed test data
test_protein = pd.read_csv('../dataPubChem/pubchem_test_protein.csv').to_numpy()
test_drug_pubchem = pd.read_csv('../dataPubChem/pubchem_test_drug.csv').to_numpy()
test_label = pd.read_csv('../dataPubChem/pubchem_y_test.csv').to_numpy()