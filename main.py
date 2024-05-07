import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib.pyplot as plt
import numpy as np

def display_images_with_text(images, texts, num_columns=5, figsize=(12, 3)):
    """
    Display multiple images in a single plot with corresponding texts above each image.

    Args:
    - images (list): List of images to display.
    - texts (list): List of corresponding texts for each image.
    - num_columns (int): Number of columns for the grid layout. Default is 5.
    - figsize (tuple): Size of the figure (width, height) in inches. Default is (12, 3).

    This method was provided by ChatGPT (https://github.com/ChatGPT)
    """
    # Function to display image and add text above it
    def display_image_with_text(image, text, ax):
        ax.imshow(image)
        ax.axis('off')  # Hide axis
        ax.set_title(text, fontsize=10)  # Add title as text above image

    # Calculate number of rows for the grid layout
    num_images = len(images)
    num_rows = int(np.ceil(num_images / num_columns))

    # Create a big figure for all images
    fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)

    # Flatten the 2D array of axes to make it easier to iterate
    axes = axes.flatten()

    # Iterate over each subplot and display image with text
    for i, (image, text) in enumerate(zip(images, texts)):
        display_image_with_text(image, text, axes[i])

    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()



def get_image(xtr, batch_id, image_id):
    try:
        batch = xtr[batch_id]
        images = batch[b'data']
        image = images[image_id]
        image_reshaped = np.array(image).reshape(3, 32, 32).transpose(1, 2, 0)  # transpose changes RGB to GBR
        return image_reshaped
    except IndexError:
        print(f"Access Error in Batch {batch_id}, id {image_id}")
        return None


def show_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis
    plt.show()


def load_data(learn_url, test_url):
    xtr = []
    for x in range(1, 6):
        data_path = str(learn_url) + str(x)
        with open(data_path, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        xtr.append(data)
    y_path = test_url
    with open(y_path, 'rb') as fo:
        y = pickle.load(fo, encoding='bytes')
    return xtr, y


def k_nearest_neighbor(xtr, test_image, k=1):
    # Find the k nearest neighbours from xtr to the test_image
    distances = []
    for i, train_batch in enumerate(xtr):
        for train_image in train_batch[b'data']:
            distances.append((distance_l1(test_image, train_image)))
    distances.sort()
    neighbors = distances[:k]  # Select only the lowest 'k' neighbours
    return neighbors


def distance_l1(image1, image2):
    return np.sum(np.abs(image1 - image2))


def distance_l2(image1, image2):
    return np.sum(np.square(image1 - image2))



# How to get label:
# batch 1 , image 0
# Xtr[1][b'labels'][0]
#
#

if __name__ == '__main__':
    xtr, y = load_data("cifar-10-batches-py/data_batch_", "cifar-10-batches-py/test_batch")
    image = get_image(xtr, 0, 0)
    # show_image(image)
