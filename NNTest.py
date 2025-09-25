import torch
import torch.nn as nn # type: ignore
from Agent import NeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 4, 52, device=device)
print(f"X = {X}")
X = X >= 0.5
X = X.to(torch.float32)
print(X)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print(y_pred.item())
