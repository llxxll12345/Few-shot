import  torch.utils.data as data
import  os
import  os.path
import  errno
from torchvision import transforms
from six.moves import urllib
import zipfile


# Omniglot with download integrated.
class Omniglot(data.Dataset):
    data_urls = {
        "train": "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
        "test":  "https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip"
    }

    def __init__(self, root_folder, data_type, download=False):
        self.root_folder = root_folder
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_type = data_type
        if not self.folder_exists():
            if download:
                self.download()
            else:
                print("Dataset not found.")

        self.files = self.get_files()
        self.class_idx = self.get_class()
        self.labels = self.get_labels()
        #print(self.class_idx)
        #print(self.labels[100:])

    def get_labels(self):
        return [self.class_idx[f[1]] for f in self.files]

    def get_files(self):
        files = []
        for (root, _, fs) in os.walk(self.root_folder):
            for f in fs:
                #print(f)
                if (f.endswith("png")):
                    r = root.split('/')
                    # language/character
                    class_name = r[-2] + "/" + r[-1]
                    files.append((f, class_name, root))
        print("Found {} files".format(len(files)))
        return files


    def get_class(self):
        class_idx = {}
        for f in self.files:
            class_name = f[1]
            if class_name not in class_idx:
                class_idx[class_name] = len(class_idx)
        print("Found {} classes".format(len(class_idx)))
        return class_idx


    def folder_exists(self):
        return os.path.exists(os.path.join(self.root_folder, "images_evaluation")) and \
            os.path.exists(os.path.join(self.root_folder, "images_background"))


    def download(self):        
        if self.folder_exists():
            return
        if not os.path.exists(os.path.join(self.root_folder)):
            os.mkdir(self.root_folder)
       
        url = self.data_urls[self.data_type]
        print('Downloading {}...'.format(url))
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]

        path = os.path.join(self.root_folder, filename)
        with open(path, 'wb') as f:
            f.write(data.read())
        
        print("Unzipping from {} to {}".format(path, self.root_folder))
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(self.root_folder)
        zip_ref.close()

        os.remove(path)
        print("Download finished.")


    def __getitem__(self, index):
        filename = self.files[index][0]
        class_name = self.files[index][1]
        root = self.files[index][2]
        img_path = str.join('/', [root, filename])
        label = self.class_idx[class_name]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


    def __len__(self):
        return len(self.files)


def test():
    # Transformer
   omniglot = Omniglot("dataset/images_background", "train", True)

test()