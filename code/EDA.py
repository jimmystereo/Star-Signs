import pandas as pd
import pyarrow.parquet as pq

file_path = 'dataset/parsed_data_public.parquet'
df = pd.read_parquet(file_path, engine='pyarrow')
df.columns[323]

df = pd.read_csv('dataset/lovoo_v3_users_instances.csv')
df.shape


df = pd.read_csv('dataset/lovoo_v3_users_api-results.csv')
df

df.columns[-10:-1]

df2 = pd.read_csv('dataset/HCMST_ver_3.04.csv')
df2


df3 = pd.read_csv('dataset/okcupid_profiles.csv')
df3.shape
df3.columns
c = [f'essay{i}' for i in range(10)]
c.remove('essay8')
c.remove()
c2  = [f'essay{i}' for i in range(3)]

df3.dropna(subset = c2).shape
