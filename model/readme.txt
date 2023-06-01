(SparseVector):
OneDrugModel.py 模型是：---》针对离散的drug编码模型
1. 蛋白质protein-并行CNN架构
2. drug是 双向GRU+并行CNN架构）
3. 特征合并后 ---》用双向GRU进行分类预测

(DenseVctor):
OneDrugModel_without_encoding.py 模型是：---》针对密集的drug编码模型
1. 蛋白质protein-并行CNN架构
2. drug是 直接使用现有的密集编码(如：Mol2vec，MPNN，Rdkit2D)特征向量，不使用CNN等模型再进行特征提取。
3. 特征合并后 ---》用双向GRU进行分类预测

TwoDrugModel.py 模型是：支持两种以上的模型输入，第一种模型通常是离散模型，第二种模型为密集模型。
1. 蛋白质protein-并行CNN架构
2. 第一种drug编码通常为离散编码模型，第二种是稠密向量。
3. 特征合并后 ---》用双向GRU进行分类预测。

TwoDrugModel_with_BiRNN.py 模型是：---》可以同时混合输入两种drug编码模型：针对两种drug编码模型。
1. 蛋白质protein-并行CNN架构
2. 第一种drug编码通常为离散编码模型，第二种可以是密集，也可以是离散编码模型，并最终用一个双向GRU来提取特征。
3. 特征合并后 ---》用双向GRU进行分类预测。

CNNCNN_OneDrugModel.py
1. 蛋白质protein-并行CNN架构
2. drug采用CNN 架构提取特征
3. 特征合并后采用 DNN进行分类预测

