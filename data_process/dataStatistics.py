import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


drug = pd.read_excel('../data/data_drug.xlsx')
protein = pd.read_excel('../data/data_protein.xlsx')

# dti = pd.read_excel('../data/data_dti.xlsx')

drugSmiles = pd.DataFrame(drug['SMILES'].unique())
drugLens = drugSmiles[0].apply(lambda x: len(str(x)))
print('drug的偏度与峰度为：')
print(drugLens.skew())
print(drugLens.kurt())

print(drugLens.describe())



proteinSeq =pd.DataFrame(protein['Sequence'].unique())
proteinSeqLens = proteinSeq[0].apply(lambda x: len(str(x)))
# ditUni = dti.iloc[:,1:4]
print('---------------------------------------------')
print(proteinSeqLens.describe())
print('protein的偏度与峰度为：')
print(proteinSeqLens.skew())
print(proteinSeqLens.kurt())

fig = plt.figure(figsize=(14,5))
ax1 = plt.subplot(1,2,1)

sns.distplot(drugLens,rug = True,
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "g"},  # 设置箱子的风格、线宽、透明度、颜色,风格包括：'bar', 'barstacked', 'step', 'stepfilled'
             kde_kws={"color": "r", "linewidth": 1, "label": "drug SMILE string length",'linestyle':'--'},   # 设置密度曲线颜色，线宽，标注、线形
             rug_kws = {'color':'r'}, axlabel='(a) SMILE length distribution' )  # 设置数据频率分布颜色

ax1 = plt.subplot(1,2,2)
sns.distplot(proteinSeqLens,rug = True,
             hist_kws={"histtype": "step", "linewidth": 1,"alpha": 1, "color": "g"},  # 设置箱子的风格、线宽、透明度、颜色,风格包括：'bar', 'barstacked', 'step', 'stepfilled'
             kde_kws={"color": "r", "linewidth": 1, "label": "protein string length",'linestyle':'--'},   # 设置密度曲线颜色，线宽，标注、线形
             rug_kws = {'color':'r'}, axlabel='(b) Protein string length distribution' )  # 设置数据频率分布颜色

# plt.savefig('str_dist.tif',dpi=600)
plt.savefig('stateps.eps', format='eps')
plt.show()

print('stop')