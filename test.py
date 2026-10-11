import torch
import torch.nn as nn
from model import LSTM_FCN
from preprocessing import test_loader
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )

model=LSTM_FCN().to(device=device)
model.load_state_dict(torch.load("lstmfcn_model.pth"))
model.eval()

all_pred=[]
all_labels=[]

with torch.no_grad():
    for inputs,labels in test_loader:

        inputs,labels=inputs.to(device),labels.to(device)
        output=model(inputs)
        _,predicted=torch.max(output,1)

        all_pred.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_pred)
acc= accuracy_score(all_labels,all_pred)

print(f"Accuracy: {acc:.2%}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_pred,
    target_names=["Normal", "Faulty"],
    zero_division=0
))

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Faulty"],
    yticklabels=["Normal", "Faulty"]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LSTM-FCN Confusion Matrix")
plt.tight_layout()
plt.show()