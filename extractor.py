import numpy as np
import pickle

def load_data(learn_url, test_url):
    xtr = []
    for x in range(1, 6):
        data_path = learn_url + str(x)
        with open(data_path, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        xtr.append(data)
    y_path = test_url
    with open(y_path, 'rb') as fo:
        y = pickle.load(fo, encoding='bytes')
    return xtr, y


def distance_l1(image1, image2):
    return np.sum(np.abs(image1 - image2))


def distance_l2(image1, image2):
    return np.sum(np.square(image1 - image2))


def create_xtr_image_dictionary(xtr):
    image_dict = {}
    id_counter = 0
    for batch in xtr:
        for i in range(len(batch[b'filenames'])):
            filename = batch[b'filenames'][i].decode('utf-8')
            label = batch[b'labels'][i]
            image_data = batch[b'data'][i]
            image_id = id_counter
            image_dict[image_id] = {'label': label, 'data': image_data}
            id_counter += 1
    return image_dict


def create_image_dictionary_from_y(y):
    image_dict = {}
    for i, label in enumerate(y[b'labels']):
        image_id = i + 1
        image_dict[image_id] = {'label': label, 'data': y[b'data'][i]}
    return image_dict


learn_url = "cifar-10-batches-py/data_batch_"
test_url = "cifar-10-batches-py/test_batch"
xtr, y = load_data(learn_url, test_url)
image_dict_xtr = create_xtr_image_dictionary(xtr)
image_dict_from_y = create_image_dictionary_from_y(y)


print(image_dict_xtr[0]['label'])
print(image_dict_xtr[1]['data'])
print(image_dict_from_y[1]['label'])

