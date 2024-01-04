# -*- coding: utf-8 -*-
"""aaaa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G1HXAO9GPkzBalmWxIrR4wRktxVkIlgL
"""

!#pip install chembl_webresource_client

from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#get the list of ids
file = open('E2012_Knime_filtered_subfulll_without=.txt')
IDs = []
for line in file:
    IDs.append(line.split(',')[1])
IDs = IDs[1:]

#getting the molecule describtors
molecule = new_client.molecule
mols = molecule.filter(molecule_chembl_id__in=IDs).only(['molecule_chembl_id','molecule_properties'])
columns_list = list(mols[1].keys())
df_mol = pd.DataFrame([mol for mol in mols],  columns=columns_list)
#expanding the features from the molecule_properties dictionary
df_pro = pd.DataFrame(columns= ['alogp',	'aromatic_rings',	'cx_logd'	,'cx_logp',	'cx_most_apka',	'cx_most_bpka'	,
                              'full_molformula'	,'full_mwt'	,'hba'	,'hba_lipinski'	,'hbd',	'hbd_lipinski'	,'heavy_atoms'	,
                              'molecular_species',	'mw_freebase',	'mw_monoisotopic',	'np_likeness_score'	,'num_lipinski_ro5_violations',
                              'num_ro5_violations',	'psa',	'qed_weighted'	,'ro3_pass',	'rtb'])
for row in range(df_mol.shape[0]):
  for key, value in df_mol.loc[row,'molecule_properties'].items():
    df_pro.loc[row,f"{key}"] = value
#merging the created dataframes
df_mol = pd.concat([df_mol,df_pro], axis = 1)
df_mol.drop('molecule_properties',axis = 1, inplace = True)
#setting index to molecule_chembl_id
df_mol.set_index("molecule_chembl_id", inplace=True)
df_mol

activity = new_client.activity
act = activity.filter(molecule_chembl_id__in=IDs)
# Extract keys as column names
columns = list(act[1].keys())
#print(df_expanded['molecule_chembl_id'] )

df_act = pd.DataFrame([d for d in act], columns=columns)
df_act_nd = df_act.drop_duplicates(subset='molecule_chembl_id')
df_act_nd['index'] = pd.RangeIndex(start=0, stop=223)  # 223 to include 222 as the last index
df_act_nd.set_index('index', inplace=True)
#dropping unnecessary columns
df_act_imp = df_act_nd.drop(['action_type', 'activity_comment',  'activity_id', 'activity_properties', 'assay_chembl_id',
'assay_description','assay_type', 'assay_variant_accession', 'assay_variant_mutation', 'bao_endpoint', 'bao_format',
'bao_label', 'canonical_smiles', 'data_validity_comment',  'data_validity_description', 'document_chembl_id','document_journal',
'document_year', 'molecule_pref_name', 'parent_molecule_chembl_id','potential_duplicate','target_tax_id', 'text_value' , 'toid',
  'uo_units', 'upper_value' ,'target_organism', 'qudt_units','record_id' , 'relation', 'src_id',  'standard_flag',
                             'standard_relation', 'standard_text_value','standard_upper_value','target_chembl_id' ], axis = 1)
#expanding ligand_efficiency
df_g = pd.DataFrame(columns = ['bei', 'le', 'lle' ,'sei'  ])
dc = {}
for row in range(df_act_imp.shape[0]):
  if df_act_imp.loc[row,'ligand_efficiency'] != None:
    #print(df_act_imp.loc[row,'ligand_efficiency'])
    dc[df_act_imp.loc[row,'molecule_chembl_id']] = list(df_act_imp.loc[row,'ligand_efficiency'].values())
print(dc)
for i in dc:
  df_g.loc[i] = dc[i]
#setting index to IDs names to be able to join the dfs
df_act_imp.set_index('molecule_chembl_id', inplace=True)
df_1 = df_g.reindex(IDs)
df_2 = df_mol.reindex(IDs)
df_3 = df_act_imp.reindex(IDs)
df_combined = pd.concat([df_1,df_2,df_3], axis=1)
df_combined.dropna(inplace=True)
#dropiing unnecessary categorical data
df_combined.drop(['full_molformula','molecular_species','ligand_efficiency','standard_type','standard_units','target_pref_name','type','units','value','ro3_pass'], axis = 1, inplace = True)

#normalization
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sd_df = pd.DataFrame(scaler.fit_transform(df_combined))
sd_df.columns =  df_combined.columns
sd_df.index =  df_combined.index
