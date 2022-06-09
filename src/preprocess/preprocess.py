import numpy as np
import os
import nibabel as nib
import torch
from torch.utils import data


def loadNiftiImage(filePath):
    img = nib.load(filePath)
    niiArray = img.get_fdata()
    dtype = img.get_data_dtype()
    return niiArray, dtype


def getFilePaths(dataFolderPath):
    ptPaths = []
    ctPaths = []
    labelPaths = []

    for root, dirs, files in os.walk(dataFolderPath):
        for file in files:
            if file.split('.')[0][-2:] == 'pt':
                ptPaths.append(os.path.join(root, file))
            elif file.split('.')[0][-2:] == 'ct':
                ctPaths.append(os.path.join(root, file))
            elif file.split('.')[0][-4:] == 'gtvt':
                labelPaths.append(os.path.join(root, file))
    return ptPaths, ctPaths, labelPaths


ptPaths, ctPaths, labelPaths = getFilePaths(
    'C:/Users/frank/Documents/GitHub/hecktor/data/hecktor2021')

# inputDtype = loadNiftiImage(ptPaths[0])[1]
# targetDtype = loadNiftiImage(labelPaths[0])[1]


class HecktorDataset(data.Dataset):
    def __init__(self,
                 inputs,
                 targets,
                 transform=None
                 ):
        self.inputs = ptPaths
        self.targets = labelPaths
        self.transform = transform
        self.inputDtype = torch.float32
        self.targetDtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        inputPath = self.inputs[index]
        targetPath = self.targets[index]

        # Load input and target
        x, y = loadNiftiImage(inputPath)[0], loadNiftiImage(targetPath)[0]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(
            self.inputDtype), torch.from_numpy(y).type(self.targetDtype)

        return x, y


trainingDataset = HecktorDataset(inputs=ptPaths,
                                 targets=labelPaths,
                                 transform=None)

trainingDataloader = data.DataLoader(dataset=trainingDataset,
                                     batch_size=2,
                                     shuffle=True)

x, y = next(iter(trainingDataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
