import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

class Dataset:

    def __init__(self, input_path, mask_path):
        self.input_path = input_path
        self.mask_path= mask_path

    def transform(self, full_input_path, full_mask_path, mask=False):
        if mask:
            # convert target from RGB to Black-white
            img = np.array(Image.open(full_mask_path).convert("L"), dtype=np.float32)
        else:
            img = np.array(Image.open(full_input_path).convert("RGB"), dtype=np.float32)
        return img

    def make_tensor(self, img_num, small_img_num):
        input_imgs = []
        mask_imgs = []
        for i in range(1, img_num+1):
            for j in range(1, small_img_num+1):
                full_input_path = self.input_path + 'img'+str(i) + '/img'+str(i)+'_'+str(j)+'.png'
                full_mask_path = self.mask_path + 'mask'+str(i) + '/mask'+str(i)+'_'+str(j)+'.png'
                img = self.transform(full_input_path, full_mask_path, mask=False)
                mask = self.transform(full_input_path, full_mask_path, mask=True)
                input_imgs.append(img)
                mask_imgs.append(mask)

        # convert from  (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]

        input_imgs = torch.FloatTensor(np.array(input_imgs) / 255).permute(0, 3, 1, 2)
        mask_imgs = torch.FloatTensor(np.array(mask_imgs))
        return input_imgs, mask_imgs

    def concat(self, inputs, masks):
        data = TensorDataset(inputs, masks)
        return data

    def k_fold(self, data, split_size):
        kf = KFold(n_splits=split_size, shuffle=True)
        train_indices = []
        val_indices = []

        for train_index, val_index in kf.split(data):
            train_indices.append(train_index)
            val_indices.append(val_index)

        return train_indices, val_indices

    def data_loader(self, data, train_index, val_index, batch_size):

        train_loader = DataLoader(dataset=data,
                                  batch_size=batch_size,
                                  sampler=train_index,
                                  shuffle=True)
        val_loader = DataLoader(dataset=data,
                                batch_size=batch_size,
                                sampler=val_index,
                                shuffle=False)

        return train_loader, val_loader