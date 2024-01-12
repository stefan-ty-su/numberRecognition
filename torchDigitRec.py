import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from sklearn.model_selection import train_test_split

device = "cuda"
print(device)

# Seeding
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Preparing Data
data = np.genfromtxt("data/custom.csv", delimiter=",")
X = torch.from_numpy(data[1:,1:]).type(torch.float)/255
y = torch.from_numpy(data[1:,0]).type(torch.LongTensor)

# Reshaping into image
X = X.reshape(X.shape[0], 28, 28).unsqueeze(dim=1)
X = torch.permute(X, (0, 1, 3, 2))
image, label = X[8], y[8]
plt.imshow(image.squeeze())
plt.show()
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

def acc_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

class MNISTModel(nn.Module):

    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_features, out_channels = hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )    

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units*49, out_features = output_features)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)

model = MNISTModel(1, 10, 10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(params = model.parameters(), lr = 0.1)

epochs = 1000
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train)
    loss = loss_fn(y_logits, y_train)
    y_pred = y_logits.argmax(dim=1)
    acc = acc_fn(y_pred, y_train)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if epoch%100 == 0:
        # print(f"Train Loss: {loss:.4f}")
        model.eval() # Evaluation mode
        with torch.inference_mode(): # Inference mode allows for more efficient computations
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_acc = acc_fn(test_pred, y_test)
        print(f"Train Loss: {loss:.4f}, Train Acc: {acc:.2f} | Test Loss: {test_loss: .4f}, Test Acc: {test_acc:.2f}")

