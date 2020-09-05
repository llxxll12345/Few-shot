import os
from PIL import Image
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms
# Folder: images_background
# Structure: root/language/character/image

TRAIN_DIR = 'omniglot/images_background'
TEST_DIR = 'omniglot/images_evaluation'
DS_STORE = '.DS_Store'

class OmiglotSet(Dataset):
    def __init__(self, dir_type):
        self.labelSet = set()
        self.label = []
        self.data = []

        dir_name = TRAIN_DIR if dir_type == 'train' else TEST_DIR
        langs = [os.path.join(dir_name, x) for x in os.listdir(dir_name) if x != DS_STORE and os.path.isdir(x)]
        for lang in langs:
            chars = [os.path.join(lang, x) for x in os.listdir(lang) if x != DS_STORE and os.path.isdir(x)]
            for ch in chars:
                imgs = [os.path.join(ch, x) for x in os.listdir(ch) if x != DS_STORE and (x.endswith('.jpg') or x.endswith('.png'))]
                if ch not in self.labelSet:
                    self.labelSet.add(ch)
                self.label.extend([len(self.labelSet) - 1] * len(imgs))
                self.data.extend(imgs)

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def test():
    dataset = OmiglotSet('test')
    train_loader = DataLoader(dataset=dataset, num_workers=8, pin_memory=True)
    for i, batch in enumerate(train_loader, 1):
        print(i, batch)


test()