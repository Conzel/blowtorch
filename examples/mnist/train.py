import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from models.models import MnistClassifier
import os

######### load data #########
def load_data():
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return train_dataloader, test_dataloader, test_data


######### training ##########
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            print(X.shape)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    ######## hyperparameters ########
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5
    save_test = True
    model_path = "model.pt"

    ######## model, optimizer, data ########
    train_dataloader, test_dataloader, test_data = load_data()
    model = MnistClassifier()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ######## training loop ########
    if not os.path.exists(model_path):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
            torch.save(model, model_path)
        print("Done!")

        ######## save model ########
        torch.save(model, model_path)
        print("Model is saved")

    ####### inference ##########
    if save_test:
        assert os.path.exists(model_path)
        model = torch.load(model_path)
        model.eval()

        # test_loop(test_dataloader, model, loss_fn)
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = test_data[sample_idx]
        pred = model(img.unsqueeze(0)).argmax(1).item()
        os.makedirs("examples", exist_ok=True)
        npimg = img.numpy()
        import matplotlib.pyplot as plt

        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("examples/example_{}.png".format(label))
        np.save("examples/example_{}.npy".format(label), img.numpy())
        print("Saved figure and numpy array.")
