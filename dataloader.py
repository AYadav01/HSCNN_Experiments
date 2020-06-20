import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import csv

class LungDataset(Dataset):

    """CSV dataset"""
    def __init__(self, train_file, num_tasks=3, transform=None):
        """
        Args:
            train_file (string): CSV file with training labels
            test_file (string, optional): CVS file with testing labels
        """
        self.train_file = train_file
        self.transform = transform
        self.num_tasks = num_tasks  # 3
        # parse the provided train file
        try:
            with self._open_for_csv(self.train_file) as file:
                # extracted data
                self.image_data = self._read_labels(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV train file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

    """Returns file reader object"""
    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    """Returns totoal number of rows in csv"""
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        label = self.load_labels(idx)
        sample = {'img': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    """Reads image and returns numpy array"""
    def load_image(self, image_index):
        img = np.load(self.image_names[image_index])
        return img.astype(np.float32)
        # return img.astype(np.float32)/255.0

    """Loads labels for given images from the dictionary created by '_read_labels()' function"""
    def load_labels(self, image_index):
        # get num of labels
        num_labels = self.num_tasks + 1  # 4
        # print("num_labels:", num_labels)
        # Gives a list of dictionary with keys as 'label1' and values as value for that label (0)
        label_list = self.image_data[self.image_names[image_index]]
        # print("label list for {} is {}".format(self.image_names[image_index], label_list))
        labels = np.zeros((0, num_labels))
        # parse labels
        for idx, a in enumerate(label_list):
            label = np.zeros((1, num_labels))
            for i in range(num_labels):
                label[0, i] = a['label' + str(i + 1)]

            labels = np.append(labels, label, axis=0)
        return labels

    """
    Returns labels dictionaryfor a particular image (CHANGES FROM ORIGINAL CODE)
    ------------
    FORMAT: {file_path: [{label0: first_col}, {"label1": second_col}, {"label2": third_col}, {"label3": forth_col}...]}
    """
    def _read_labels(self, csv_reader):
        result = {}
        num_cols = self.num_tasks + 2
        for line, col in enumerate(csv_reader):
            line += 1
            label_dict = {}
            try:
                img_file = col[0]
                # print("label4", img_file) # gets the last label: malignancy score
                for i in range(1, len(col)):
                    label_dict['label' + str(i)] = col[i]
                # print(label_dict)
            except ValueError:
                raise_from(ValueError(
                    'line {}: format should be \'label1,label2,label3,label4, img_file\' or \',,,,img_file\''.format(
                        line)), None)
            # If label is not in dictionary, create a key with empty list as dict values
            if img_file not in result:
                result[img_file] = []
            result[img_file].append(label_dict)
        return result

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, labels = sample['img'], sample['label']
        D, H, W = image.shape
        img = torch.from_numpy(image)
        img = np.transpose(img, (2, 1, 0))
        new_image = img.unsqueeze(0)
        new_labels = torch.from_numpy(labels)
        return {'img': new_image, 'label': new_labels}