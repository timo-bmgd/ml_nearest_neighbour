import numpy as np
import matplotlib.pyplot as plt
import pickle
def get_image_vector(xtr, batch_idx, image_idx):
    try:
        batch = xtr[batch_idx]
        images = batch[b'data']
        image_vector = images[image_idx]
        return image_vector
    except IndexError:
        print("batch_index or image_index not correct, or maybe you're just a failure and your parents never loved you")
        return None


def reshape_image(image_data):
    # Reshape the image data into a 32x32x3 array
    image_reshaped = np.array(image_data).reshape(3, 32, 32).transpose(1, 2, 0)  # transpose changes RGB to GBR

    return image_reshaped


def visualize_image(image_reshaped):
    # Visualize the image
    plt.imshow(image_reshaped)
    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == '__main__':
    Xtr = []

    for x in range(1, 6):
        data_path = "cifar-10-batches-py/data_batch_" + str(x)
        with open(data_path, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        Xtr.append(data)
    print(len(data))
    y_path = "cifar-10-batches-py/test_batch"
    with open(y_path, 'rb') as fo:
        y = pickle.load(fo, encoding='bytes')
    print(f"It is {y}")
    image_vector = get_image_vector(Xtr, 0, 0)

    image = reshape_image(image_vector)

    visualize_image(image)
