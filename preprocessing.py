import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import torch
from torch.utils.data import TensorDataset, DataLoader

# Verileri oku
df = pd.read_csv("secom.data", header=None, sep=r"\s+")
labels = pd.read_csv("secom_labels.data", header=None, sep=r"\s+")

df["label"] = labels[0].values

# Eksik değer oranı %90'dan fazla olan sütunları bul
cols = df.drop(columns=["label"]).columns[
    df.drop(columns=["label"]).isnull().mean() > 0.90
]

# Bu sütunları kaldır
df.drop(columns=cols, inplace=True)

# Kalan eksik değerleri sütun medyanıyla doldur
df.iloc[:, :-1] = df.iloc[:, :-1].fillna(
    df.iloc[:, :-1].median()
)

# Etiketleri -1 ve 1'den 0 ve 1'e dönüştür
df["label"] = df["label"].apply(lambda x: 0 if x == -1 else 1)

# Özellikler ve etiketler
x = df.drop(columns=["label"]).values
y = df["label"].values

# Önce eğitim ve test verilerini ayır
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Ölçekleyiciyi yalnızca eğitim verisinde öğren
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# SMOTE'u yalnızca eğitim verisine uygula
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Tensor oluştur
x_train_tensor = torch.tensor(
    x_train, dtype=torch.float32
)
y_train_tensor = torch.tensor(
    y_train, dtype=torch.long
)

x_test_tensor = torch.tensor(
    x_test, dtype=torch.float32
)
y_test_tensor = torch.tensor(
    y_test, dtype=torch.long
)

# Dataset oluştur
train_dataset = TensorDataset(
    x_train_tensor, y_train_tensor
)
test_dataset = TensorDataset(
    x_test_tensor, y_test_tensor
)

# DataLoader oluştur
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False
)

# Ölçekleyiciyi kaydet
joblib.dump(scaler, "scaler.pkl")

print("Preprocessing tamamlandı!")
print("Eğitim örnek sayısı:", len(train_dataset))
print("Test örnek sayısı:", len(test_dataset))
print("Özellik sayısı:", x_train.shape[1])
