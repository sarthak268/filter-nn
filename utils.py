import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

def save_image_grid(images, directory_path, img_type):
    # img_size = images.size(2)
    # batch_size = images.size(0)
    # grid_images = images.cpu().detach().view(8, batch_size//8, 1, img_size, img_size)

    # # Concatenate along the columns (dim=3) to create rows of images
    # grid_images = torch.cat([grid_images[i].squeeze(2) for i in range(grid_images.size(0))], dim=3)

    # # Concatenate along the rows (dim=2) to create the final grid
    # grid_images = torch.cat([grid_images[i] for i in range(grid_images.size(0))], dim=2)

    # # Convert the tensor to a NumPy array
    # grid_images_np = grid_images.numpy()

    # Display the grid using matplotlib
    # plt.imshow(np.transpose(grid_images_np, (1, 2, 0)), cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Save the grid as an image file (e.g., PNG)
    save_image(images, directory_path + 'image_{}.png'.format(img_type), nrow=8, normalize=True)
