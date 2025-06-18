import torch
import os
import zipfile
import io
from torch.utils.data import Dataset

class CondDataset(Dataset):
    def __init__(self, path, img_res_h, img_res_w, img_input_channels, img_cond_channels, name,
                 xflip=False, yflip=False, cache=False, max_mismatches=0, **kwargs):
        """
        Args:
            path (str): Folder or .zip file with .pt files.
            img_res_h, img_res_w (int): Height and width of image/condition arrays.
            img_channels (int): Channels in 'image' array.
            cond_channels (int): Channels in 'conditions' array.
            xflip, yflip (bool): Enable horizontal/vertical deterministic flips.
            cache (bool): Cache data in RAM.
            max_mismatches (int): Max shape mismatches to show before stopping.
        """
        self.path               = path
        self.img_res_h          = img_res_h
        self.img_res_w          = img_res_w
        self.img_input_channels = img_input_channels
        self.img_cond_channels  = img_cond_channels
        self.image_shape        = (img_input_channels, img_res_h, img_res_w)
        self.condition_shape    = (img_cond_channels, img_res_h, img_res_w)
        self.name               = name
        self._xflip             = xflip
        self._yflip             = yflip
        self._cache             = cache
        self._cached_data       = {}
        self._zipfile           = None

        # File list
        if os.path.isdir(path):
            self._type = 'dir'
            self._file_list = sorted(f for f in os.listdir(path) if f.endswith('.pt'))
        elif os.path.isfile(path) and path.endswith('.zip'):
            self._type = 'zip'
            self._file_list = sorted(f for f in zipfile.ZipFile(path, 'r').namelist() if f.endswith('.pt'))
        else:
            raise ValueError(f"Unsupported path: {path}")

        if len(self._file_list) == 0:
            raise RuntimeError(f"No .pt files found in {path}")

        # Flipping variants
        self.variants = [(False, False)]
        if xflip and yflip:
            self.variants = [(False, False), (True, False), (False, True), (True, True)]
        elif xflip:
            self.variants = [(False, False), (True, False)]
        elif yflip:
            self.variants = [(False, False), (False, True)]

        self.num_variants = len(self.variants)
        self.dataset_len = len(self._file_list) * self.num_variants

        # Validate file shapes
        self._check_shapes(max_mismatches)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        base_idx = index // self.num_variants
        variant_idx = index % self.num_variants
        apply_xflip, apply_yflip = self.variants[variant_idx]

        fname = self._file_list[base_idx]
        data = self._load_file(fname)

        image = data['image']
        cond = data['conditions']

        if apply_xflip:
            image = torch.flip(image, dims=[2])
            cond = torch.flip(cond, dims=[2])
        if apply_yflip:
            image = torch.flip(image, dims=[1])
            cond = torch.flip(cond, dims=[1])

        return image, cond

    def _load_file(self, fname):
        if self._cache and fname in self._cached_data:
            return self._cached_data[fname]

        if self._type == 'zip':
            if self._zipfile is None:
                self._zipfile = zipfile.ZipFile(self.path, 'r')
            with self._zipfile.open(fname, 'r') as f:
                buffer = io.BytesIO(f.read())
                data = torch.load(buffer)
        else:
            data = torch.load(os.path.join(self.path, fname))

        if self._cache:
            self._cached_data[fname] = data
        return data

    def _check_shapes(self, max_mismatches=0):
        print("Checking file shapes...")
        mismatch_count = 0
        for fname in self._file_list:
            try:
                data = self._load_file(fname)
                img_shape = tuple(data['image'].shape)
                cond_shape = tuple(data['conditions'].shape)

                if img_shape != self.image_shape or cond_shape != self.condition_shape:
                    print(f"Mismatch in {fname}: image {img_shape}, condition {cond_shape}")
                    mismatch_count += 1
                    if mismatch_count >= max_mismatches:
                        print("Max mismatches reached.")
                        break
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                mismatch_count += 1
                if mismatch_count >= max_mismatches:
                    break

        if mismatch_count == 0:
            print("All files match expected shapes.")
        else:
            raise ValueError(f"{mismatch_count} file(s) had shape mismatches.")

    def close(self):
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None

    # Needed to make it work with multiprocessing (e.g., num_workers > 0)
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zipfile'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._zipfile = None
