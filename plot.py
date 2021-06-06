import torchvision
import matplotlib.pyplot as plt


def make_grid(imgs, nrow):
    imgs_grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=100)
    plt.imshow(imgs_grid.permute(1, 2, 0))
    plt.show()

