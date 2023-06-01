# 得到 蛋白质编码 和 drug pubchem编码 的汇总文件
import numpy as np
import pandas as pd
from keras.preprocessing import sequence

dti = pd.read_excel("data_dti.xlsx")
# ESPF： 编码维数 2586
drug_ESPF = pd.read_csv("drug_encoding_ESPF.csv",index_col="Compound_ID")
drug_ESPF.columns = drug_ESPF.columns.map(lambda x:x+'_espf')
# pubchem:编码维数：881
drug_Pubchem = pd.read_csv("drug_encoding_Pubchem.csv",index_col="Compound_ID")
drug_Pubchem.columns = drug_Pubchem.columns.map(lambda x:x+'_pubchem')

# drug = pd.merge(drug_ESPF,drug_Pubchem, on="Compound_ID",suffixes=('_de', '_dp'))
drug = pd.merge(drug_ESPF,drug_Pubchem, on="Compound_ID")

protein = pd.read_csv("data_proteinID_encode.csv",index_col="Protein_ID")
protein.columns = protein.columns.map(lambda x:x+'_pro')

dti_df_combine = pd.merge(dti, protein, left_on="Protein_ID", right_index=True)
dti_df_combine2 = pd.merge(dti_df_combine, drug, left_on="Compound_ID", right_index=True)
# 依次是：蛋白质[0_pro-->2499_pro]  drug_ESPF[0_espf--->2585_espf]  drug_Pubchem[0_pubchem--->880_pubchem] Lable[Lable]
total_protein_drug_Pubchem_Lable = pd.concat([dti_df_combine2.loc[:,'0_pro':'2499_pro'],
                                              dti_df_combine2.loc[:,'0_espf':'2585_espf'], dti_df_combine2.loc[:,'0_pubchem':'880_pubchem'], dti_df_combine2['Label']],axis=1)

total_protein_drug_Pubchem_Lable.to_csv("total_protein_drug_ESPF_and_Pubchem_Lable.csv",index=None)
print("stop")