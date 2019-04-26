from sklearn.model_selection import train_test_split
import os
import subprocess
import pandas as pd
from string import digits

images_dir = 'labelled_images'
path_to_data = 'build/darknet/x64/data/obj/'

t = os.listdir(images_dir )
t = [i for i in t if i.endswith('.png')]

df = pd.DataFrame({'images':t})
remove_digits = str.maketrans('', '', digits)
df['group']= df['images'].apply(lambda x: x.translate(remove_digits).replace('.png', ''))

train_df, test_df = train_test_split(df, stratify = df['group'], test_size = 0.1)


with open('train.txt', 'w') as f:
    for i in train_df['images'].tolist():
        f.write(path_to_data + "/" + i + '\n')

with open('test.txt', 'w') as f:
    for i in test_df['images'].tolist():
        f.write(path_to_data + "/" + i + '\n')




