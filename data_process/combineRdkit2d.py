# 得到 蛋白质编码 和 drug Daylight编码 的汇总文件
import numpy as np
import pandas as pd
from keras.preprocessing import sequence

dti = pd.read_excel("data_dti.xlsx")

drug = pd.read_csv("drug_encoding_rdkit_2d_normalized.csv",index_col="Compound_ID")
protein = pd.read_csv("data_proteinID_encode.csv",index_col="Protein_ID")

dti_df_combine = pd.merge(dti, protein, left_on="Protein_ID", right_index=True)
dti_df_combine2 = pd.merge(dti_df_combine, drug, left_on="Compound_ID", right_index=True)
total_protein_drug_rdkit_2d_Lable = pd.concat([dti_df_combine2.loc[:,'0_x':'2499'],dti_df_combine2.loc[:,'0_y':'199_y'],dti_df_combine2['Label']],axis=1)

total_protein_drug_rdkit_2d_Lable.to_csv("total_protein_drug_rdkit_2d_Lable.csv",index=None)
print("stop")