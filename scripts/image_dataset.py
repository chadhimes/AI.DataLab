import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class PlayImageDataset(Dataset):
    """PyTorch Dataset for images in `out_all_plays` with yardage labels.

    The labels CSV should have at least columns: `img_name` and `yards_gained`.
    """

    def __init__(self, images_dir, labels_csv, transform=None, filter_missing=True, label_transform=None):
        self.images_dir = images_dir
        self.transform = transform
        # optional callable to transform numeric label (e.g., normalization)
        # Avoid non-picklable callables for multiprocessing; prefer numeric attributes below.
        self.label_transform = None
        self.label_mean = None
        self.label_std = None

        # load labels
        self.df = pd.read_csv(labels_csv)

        # Accept both `img_name` and `img` common variants
        if 'img_name' not in self.df.columns and 'img' in self.df.columns:
            self.df = self.df.rename(columns={'img': 'img_name'})

        if filter_missing:
            # only keep rows with an existing image file
            self.df['image_path'] = self.df['img_name'].apply(lambda x: os.path.join(images_dir, x))
            self.df = self.df[self.df['image_path'].apply(os.path.exists)].reset_index(drop=True)
        else:
            self.df['image_path'] = self.df['img_name'].apply(lambda x: os.path.join(images_dir, x))

        # ensure yard label column exists
        if 'yards_gained' not in self.df.columns:
            raise ValueError('labels csv must contain `yards_gained` column')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = float(row['yards_gained'])
        # Prefer numeric mean/std for picklable normalization when using multiple workers
        if self.label_mean is not None:
            std = self.label_std if (self.label_std is not None and self.label_std != 0) else 1.0
            label = (label - self.label_mean) / std
        elif self.label_transform is not None:
            label = self.label_transform(label)
        return img, label


def collate_fn(batch):
    # default collate from torch works fine; keep placeholder if future changes needed
    imgs, labels = zip(*batch)
    return imgs, labels
