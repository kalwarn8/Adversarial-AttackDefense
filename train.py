import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import preprocessing
from models.model import MLP


X_train, X_test, y_train, y_test, feature_names = preprocessing.load_data("data/adult.data")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

input_size = X_train.shape[1]

model = MLP(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200

for epoch in range(epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")

print("Training complete")