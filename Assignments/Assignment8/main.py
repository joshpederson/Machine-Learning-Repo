import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# Custom MNIST Dataset
class MNIST(Dataset):
    def __init__(self):
        # Load from Kaggle: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
        # Download dataset too large to upload to git
        df = pd.read_csv("mnist_train.csv")
        self.train_numbers = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 28, 28)
        self.train_labels = torch.tensor(df["label"].to_numpy())

        df = pd.read_csv("mnist_test.csv")
        self.test_numbers = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 28, 28)
        self.test_labels = torch.tensor(df["label"].to_numpy())

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_numbers[item], self.train_labels[item]

    def __len__(self):
        return self.len


# CNN Model Definition
class NumberClassify(nn.Module):
    def __init__(self):
        super(NumberClassify, self).__init__()
        self.in_to_h1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))  # 32 x 28 x 28
        self.h1_to_h2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))  # 32 x 28 x 28
        self.h2_to_h3 = nn.Conv2d(32, 8, (3, 3))  # 8 x 26 x 26 -> pool -> 8 x 13 x 13
        self.h3_to_h4 = nn.Linear(8 * 6 * 6, 20)
        self.h4_to_out = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        x = F.dropout2d(x, 0.1)
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.h2_to_h3(x))
        x = F.max_pool2d(x, (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.h3_to_h4(x))
        return self.h4_to_out(x)


# Training Function
def trainNN(model, dataset, device, epochs=5, batch_size=16, lr=0.001, display_test_acc=False):
    mnist_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(mnist_loader)):
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0

        if display_test_acc:
            with torch.no_grad():
                model.eval()
                predictions = torch.argmax(model(dataset.test_numbers.to(device)), dim=1)
                correct = (predictions == dataset.test_labels.to(device)).sum().item()
                print(f"Test accuracy: {correct / len(dataset.test_labels):.4f}")


# Visualization Function
def visualize_layers(model, image, device):
    model.eval()
    activations = []

    x = image.unsqueeze(0).to(device)

    x = F.relu(model.in_to_h1(x))
    activations.append(("Conv1", x.clone().detach()))

    x = F.relu(model.h1_to_h2(x))
    x = F.dropout2d(x, 0.1)
    activations.append(("Conv2 + Dropout", x.clone().detach()))

    x = F.max_pool2d(x, (2, 2))
    activations.append(("MaxPool1", x.clone().detach()))

    x = F.relu(model.h2_to_h3(x))
    activations.append(("Conv3", x.clone().detach()))

    x = F.max_pool2d(x, (2, 2))
    activations.append(("MaxPool2", x.clone().detach()))

    for name, act in activations:
        fig, axes = plt.subplots(1, min(8, act.shape[1]), figsize=(15, 5))
        fig.suptitle(name)
        for i in range(min(8, act.shape[1])):
            axes[i].imshow(act[0, i].cpu(), cmap="gray")
            axes[i].axis('off')
        plt.show()


# Main Script
if __name__ == "__main__":
    # Prepare dataset and model
    mnist = MNIST()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    number_classify = NumberClassify().to(device)

    print(f"Total parameters: {sum(p.numel() for p in number_classify.parameters())}")

    # Train
    trainNN(number_classify, mnist, device, epochs=1, display_test_acc=True)

    # Visualize intermediate layers for a test image
    test_img = mnist.test_numbers[0]
    visualize_layers(number_classify, test_img, device)
