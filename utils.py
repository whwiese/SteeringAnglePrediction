import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image, angle):
    img = np.array(image)
    img = np.transpose(img, (1,2,0))

    #create figure, display image
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    title = "Steering Angle: "+str(angle.item())

    plt.title(title)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()

def evaluateModel(model, train_loader, val_loader, loss_fn, device="cpu"):
    """
    returns mse loss of model on train and validation sets
    """
    model.eval()

    train_loss_sum = 0.0
    val_loss_sum = 0.0

    train_batches = 0
    val_batches = 0

    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        train_loss_sum += loss.item()
        train_batches += 1

    for batch_index, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        val_loss_sum += loss.item()
        val_batches += 1

    train_mse = train_loss_sum/train_batches
    val_mse = val_loss_sum/val_batches

    model.train()

    return train_mse, val_mse

def plotMSEs(train_mses, val_mses, epochs):
    fig = plt.figure()
    ax = plt.axes()

    train = ax.plot(epochs, train_mses, color='blue', label="train")
    val = ax.plot(epochs, val_mses, color='red', label="val")

    plt.title("Training Loss vs Val Loss")
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')

    plt.legend(loc="upper right")

    plt.show()
