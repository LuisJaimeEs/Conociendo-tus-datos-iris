import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


sns.set()

df=pd.read_csv('iris.csv')

df.info()

df.describe()

print(df[df.duplicated()])

print(df["class"].value_counts())

plt.title('Elementos por clase')

sns.countplot(x=df["class"])

sns.scatterplot(x=df["sepal length"], y=df["sepal width"], hue=df["class"])

sns.scatterplot(x=df["sepal length"], y=df["petal width"], hue=df["class"])
sns.scatterplot(x=df["sepal length"], y=df["petal width"], hue=df["class"])

sns.pairplot(df,hue="class")

#separar las clases mlo mas que se pueda

plt.figure()

sns.heatmap(df.corr(), annot=True)

print(df.groupby("class").agg(["mean", "median"]))

plt.show()