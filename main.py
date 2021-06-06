from model import UNet
from data import Dataset
from train import train
from validate import validate
from save import save_img, color_mapping
from plot import make_grid

import torch
from torch import nn, utils
import torch.optim as optim
from torchvision import utils


# HYPER-PARAMETERS
load_model = False
batch_size = 2
train_size = 64
val_size = 8
split_size = 9
epochs = 5
lr = 0.005
weight_decay = 0.01
momentum = 0.9
num_class = 8
num_imgs = 72

# SET DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# PATH
mask_path = './imgs/labels/'
input_path = './imgs/inputs/'


def main():
    # MODEL
    unet = UNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unet.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # PREPARE DATA
    data = Dataset(input_path, mask_path)
    input_imgs, mask_imgs = data.make_tensor(num_imgs)
    data = data.concat(input_imgs, mask_imgs)
    train_indices, val_indices = data.k_fold(data, split_size)


    # LOAD TRAINED MODEL
    if load_model:
        unet.load_state_dict(torch.load('./checkpoint/state_dict_model.pt'))

    acc_list = []

    for i in range(len(train_indices)):
        train_loader, val_loader = data.data_loader(data, train_indices[i], val_indices[i], batch_size)

        outputs = []
        acc_per_kfold = 0

        for epoch in range(epochs):
            # TRAINING
            train(unet, epoch, optimizer, criterion, train_loader, epochs, device)

            # VALIDATION
            acc_per_epoch, outputs = validate(unet, num_class, val_loader, val_size, batch_size, device, outputs)
            acc_per_kfold += acc_per_epoch

        acc_per_kfold /= epochs
        acc_list.append(acc_per_kfold)
        print(f'K-fold {i+1} average accuracy: {acc_per_kfold}')
        print('=' * 40)

    overall_acc = sum(acc_list) / len(acc_list)

    # SAVE CHECKPOINT
    torch.save(unet.state_dict(), './checkpoint/state_dict_model.pt')

    # VISUALIZE RESULTS
    outputs_c = color_mapping(outputs)
    make_grid(outputs_c, nrow=9)

    # SAVE
    save_img(outputs_c)

if __name__ == "__main__":
    main()