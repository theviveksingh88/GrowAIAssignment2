import torch
import torch.nn as nn
import torch.optim as optim

#1Define Class
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()

        self.hidden = nn.Linear(2,4)
        self.output = nn.Linear(4,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

#2 XOR Datasets
x =  torch.tensor([[0.0,0.0],
                   [0.0,1.0],
                   [1.0,0.0],
                   [1.0,1.0]],dtype=torch.float32)

y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]], dtype=torch.float32)

#3 Model, Optimizer, BCE Loss
model = XORModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

#4 Training Loop
print("--- Starting the Training Loop")
for epoch in range(5001):
     #forward pass
    predictions = model(x)
    loss = criterion(predictions, y)

    #Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Log Loss
if epoch % 500 == 0:
    print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")


with torch.no_grad():
    final_predictions = model(x)
    for i in range(4):
        input_pair = x[i].tolist()
        expected = int(y[i].item())
        raw_pred = final_predictions[i].item()
        classified = int(round(raw_pred))

        print(f"Input: {input_pair} -> Expected: {expected} | Predicted: {raw_pred:.4f} | Classified: {classified}")





