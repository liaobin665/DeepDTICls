
import numpy as np
import pandas as pd
from keras.preprocessing import sequence

protein_encoding_len = 2500
protein_seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
protein_seq_dic = {w: i+1 for i,w in enumerate(protein_seq_rdic)}

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[each] for each in seq]



#dti = pd.read_excel("data_dti.xlsx")
#drug = pd.read_excel("drug_encoding_pubchem.xlsx",index_col="Compound_ID")
protein = pd.read_excel("data_protein.xlsx")
strlen = protein['Sequence'].apply(lambda x:len(str(x)))

print(strlen.describe())

protein['encoded_sequence'] = protein.Sequence.map(lambda a: encodeSeq(a, protein_seq_dic))
pad_protein_seq = pd.DataFrame(sequence.pad_sequences(protein["encoded_sequence"].values, protein_encoding_len))

protein = pd.concat([protein,pad_protein_seq],axis=1)

protein = pd.concat([protein['Protein_ID'],protein.iloc[:,4:2505]],axis=1)

# protein['protein_encode']=protein.iloc[:,1:2050].apply(lambda x:np.concatenate(list(x)))


protein.to_csv("data_proteinID_encode.csv")

#
# dti_df_combine = pd.merge(dti, protein, left_on="Protein_ID", right_index=True)
# dti_df_combine2 = pd.merge(dti_df_combine, drug, left_on="Compound_ID", right_index=True)
#
#
# df_out = dti_df_combine2[["SMILES","Sequence","Label"]]
# df_out.to_csv("smile_protein_lable_pair.csv",sep=" ",columns=None,index=None,header=0)
print("stop")
