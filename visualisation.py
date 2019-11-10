import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


root = './train/train/'
data = []
for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))
df = pd.DataFrame(data, columns=['class', 'file_path'])
len_df = len(df)
print(f"There are {len_df} images")

print(df['class'].value_counts())

# Figure 1
plt.figure()
df['class'].value_counts().plot(kind='bar')
plt.title('Class counts')

# Figure 1
plt.figure()
_ = sns.countplot(y=df['class'])
plt.title('Class counts')

###
plt.show()
