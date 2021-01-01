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


def testModel(model, test_loader, loss_fn, 
        title="Model vs Human Steering Wheel Angles", device="cpu"):
    """
    plots model outputs vs human steering angles, returns loss of model
    outputs and loss of always outputting 0
    """
    model.eval()

    zero_loss_sum = 0.0
    model_loss_sum = 0.0

    test_batches = 0
    model_preds = []
    human_angles = []

    for batch_index, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        model_loss = loss_fn(out, y)
        zero_loss = loss_fn(torch.zeros_like(out), y)
        model_loss_sum += model_loss.item()
        zero_loss_sum += zero_loss.item()
        test_batches += 1

        #record data for plotting
        model_preds += out.tolist()
        human_angles += y.tolist()

    zero_loss = zero_loss_sum/test_batches
    model_loss = model_loss_sum/test_batches

    #plot preds vs human angles
    fig = plt.figure()
    ax = plt.axes()

    frames = range(len(model_preds))

    preds = ax.plot(frames, model_preds,
            color='orange', label="Model"
    )
    human = ax.plot(frames, human_angles,
            color='blue', label="Human Driver"
    )

    plt.title(title)
    plt.xlabel('Frame #')
    plt.ylabel('Steering Wheel Angle (degrees)')

    plt.legend(loc="upper right")

    plt.show()

    model.train()

    return model_loss, zero_loss
