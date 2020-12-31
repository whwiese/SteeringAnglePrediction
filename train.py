import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms.functional as FT
from torch.utils.data import (DataLoader, random_split)
from model import CNNDriver
from dataset import SteeringAngleDataset
from utils import (evaluateModel, plotMSEs)
import time

#model hyperparameters
INPUT_DIMS = (3,455,256)

#other hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
EPOCHS = 200
NUM_WORKERS = 32
PIN_MEMORY = True
LOAD_CHECKPOINT = False
DROP_LAST = False
TRAIN_DATA = "drive/MyDrive/DLData/SteeringAngle/drive1_train.csv"
VAL_DATA = "drive/MyDrive/DLData/SteeringAngle/drive1_val.csv"
TEST_DATA = None 
SAVE_CHECKPOINT_PATH = "trial1/checkpoint"
LOAD_CHECKPOINT_PATH = "trial1/checkpoint"
IMG_DIR = "drive1"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

#transforms.Resize(INPUT_DIMS[1:])

transform = Compose([transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    
    mean_loss = []

    for batch_index, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE).float()
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    model = CNNDriver(input_dims=INPUT_DIMS).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss(reduction='mean') 

    epochs_passed = 0
    val_mses = []
    train_mses = []
    epochs_recorded = []

    if LOAD_CHECKPOINT:
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
          param_group['lr'] = LEARNING_RATE
        epochs_passed = checkpoint['epoch']
        train_mses = checkpoint['train_mses']
        val_mses = checkpoint['val_mses']
        epochs_recorded = checkpoint['epochs_recorded']

    train_dataset = SteeringAngleDataset(
            TRAIN_DATA, img_dir=IMG_DIR, transform=transform, 
    )

    val_dataset = SteeringAngleDataset(
            VAL_DATA, img_dir=IMG_DIR, transform=transform, 
    )

    if torch.cuda.is_available():
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
                drop_last=DROP_LAST,
        )
        
        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                shuffle=False,
                drop_last=DROP_LAST,
        )

    else:
        train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=DROP_LAST,
        )
        
        val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=DROP_LAST,
        )



    for epoch in range(EPOCHS):
        if epochs_passed%1 == 0:
            
            train_mse, val_mse = evaluateModel(model, train_loader,
                    val_loader, loss_fn, device=DEVICE
            )

            train_mses.append(train_mse)
            val_mses.append(val_mse)
            epochs_recorded.append(epochs_passed)

            plotMSEs(train_mses, val_mses, epochs_recorded)

            print("Train MSE: %f"%(train_mse))
            print("Val MSE: %f"%(val_mse))

        if epochs_passed%5 == 0:
            save_path = SAVE_CHECKPOINT_PATH + ("checkpoint_%de"%(epochs_passed))+".pt"
            checkpoint = { 
                'epoch': epochs_passed,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_mses': train_mses,
                'val_mses': val_mses,
                'epochs_mses': epochs_recorded,
            }
            torch.save(checkpoint,save_path)
            print("Trained for %d epochs"%(epochs_passed))

        train_fn(train_loader, model, optimizer, loss_fn)
        epochs_passed += 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Training time: %f seconds"%(time.time()-start_time))
