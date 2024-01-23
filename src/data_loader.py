import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class COCODataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.dataloader"""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images root directory, captions and vocabulary module

        Args:
            root: image directory
            json: coco annotations file path
            vocab: vocabulary module
            transform: image transformer
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns a pair of image and caption

        Args:
            index: index of the image
        """
        ann_id = self.ids[index]
        caption = self.coco[ann_id]['caption']
        img_id = self.coco[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)
    

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuples (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort by caption length
    data.sort(key=lambda x:len(x[1]), reverse=True)
    images, captions = zip(*data)

    #Merge 3D tensor of images to a 4D tensor
    images = torch.stack(images, 0)

    # Merge 1D tensor of captions to a 2D tensor
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in captions:
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset.
    
    Args:
        root: image directory
        json: coco annotation file
        vocab: vocabulary module
        tranform: image transformer
        batch_size: batch size
        shuffle: boolean value for data shuffling
        num_workers: number of workers

    Returns (images, captions, lengths) for each iteration.
        images: a tensor of shape (batch_size, 3, 224, 224).
        captions: a tensor of shape (batch_size, padded_length).
        lengths: a list indicating valid length for each caption. length is (batch_size).
    """

    coco = COCODataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    
    return data_loader
    

    



