import torch
import torch.nn as nn
from model import LSTM_FCN
from preprocessing import train_loader,test_loader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )

model= LSTM_FCN().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=20

for epoch in range(epochs):

    model.train()
    train_loss=0
    correct=0
    total=0

    for inputs,labels in train_loader:

        inputs,labels=inputs.to(device),labels.to(device)

        optimizer.zero_grad()
        output=model(inputs)
        loss= criterion(output,labels)
        loss.backward()
        optimizer.step()

        train_loss +=loss.item()
        _,predicted=torch.max(output,1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)

    avg_loss=train_loss/total
    accuracy=correct/total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")

torch.save(model.state_dict(),"lstmfcn_model.pth")
print("model is saved")    