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
        self.num_tasks = num_tasks
        # parse the provided train file
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_labels(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV train file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())

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

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        label = self.load_labels(idx)
        sample = {'img': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        img = np.load(self.image_names[image_index])
        return img.astype(np.float32) / 255.0

    def load_labels(self, image_index):
        num_labels = self.num_tasks + 1
        label_list = self.image_data[self.image_names[image_index]]
        labels = np.zeros((0, num_labels))
        # parse labels
        for idx, a in enumerate(label_list):
            label = np.zeros((1, num_labels))
            for i in range(num_labels):
                label[0, i] = a['label' + str(i + 1)]
            labels = np.append(labels, label, axis=0)
        return labels

    def _read_labels(self, csv_reader):
        result = {}
        num_cols = self.num_tasks + 2
        for line, col in enumerate(csv_reader):
            line += 1
            label_dict = {}
            try:
                img_file = col[-1]
                for i in range(0, len(col) - 1):
                    label_dict['label' + str(i + 1)] = col[i]
            except ValueError:
                raise(ValueError(
                    'line {}: format should be \'label1,label2,label3,label4, img_file\' or \',,,,img_file\''.format(
                        line)), None)
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