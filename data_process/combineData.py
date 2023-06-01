import numpy as np
import pandas as pd
from keras.preprocessing import sequence

smile = pd.read_excel("data_drug.xlsx")

dti = pd.read_excel("data_dti.xlsx")
drug = pd.read_excel("data_compound.xlsx",index_col="Compound_ID")
protein = pd.read_excel("data_protein.xlsx",index_col="Protein_ID")

dti_df_combine = pd.merge(dti, protein, left_on="Protein_ID", right_index=True)
dti_df_combine2 = pd.merge(dti_df_combine, drug, left_on="Compound_ID", right_index=True)

df_out = dti_df_combine2[["SMILES","Sequence","Label"]]
df_out.to_csv("smile_protein_lable_pair.csv",sep=" ",columns=None,index=None,header=0)
print("stop")
