import torch
import os
import zipfile
import io
from torch.utils.data import Dataset

class PTDataset(Dataset):
	def __init__(self, path, img_res_h, img_res_w, img_channels, cond_channels, xflip=False, yflip=False, cache=False, max_mismatches=0):
		"""
		Args:
			path (str): Path to folder or zip file containing .pt files.
			img_res_h (int): Image height.
			img_res_w (int): Image width.
			img_channels (int): Number of image channels.
			cond_channels (int): Number of condition channels.
			xflip (bool): Enable horizontal flip (dataset expansion).
			yflip (bool): Enable vertical flip (dataset expansion).
			cache (bool): Enable in-memory caching of .pt files.
			max_mismatches (int): Max shape mismatches to report before stopping.
		"""
		self.path = path
		self.image_shape = (img_channels, img_res_h, img_res_w)
		self.condition_shape = (cond_channels, img_res_h, img_res_w)
		self._xflip = xflip
		self._yflip = yflip
		self._cache = cache
		self._cached_data = {}

		# Detect input source type and list .pt files
		self._type = None
		self._zipfile = None
		if os.path.isdir(path):
			self._type = 'dir'
			self._file_list = sorted(f for f in os.listdir(path) if f.endswith('.pt'))
		elif os.path.isfile(path) and path.endswith('.zip'):
			self._type = 'zip'
			self._zipfile = zipfile.ZipFile(path, 'r')
			self._file_list = sorted(f for f in self._zipfile.namelist() if f.endswith('.pt'))
		else:
			raise ValueError(f"Unsupported path: {path}")

		if len(self._file_list) == 0:
			raise RuntimeError(f"No .pt files found in {path}")

		# Define flip variants
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
		self._check_shapes(max_mismatches=max_mismatches)

	def __len__(self):
		return self.dataset_len

	def _load_file(self, fname):
		if self._cache and fname in self._cached_data:
			return self._cached_data[fname]

		if self._type == 'dir':
			data = torch.load(os.path.join(self.path, fname))
		else:
			with self._zipfile.open(fname, 'r') as f:
				buffer = io.BytesIO(f.read())
				data = torch.load(buffer)

		if self._cache:
			self._cached_data[fname] = data
		return data

	def __getitem__(self, index):
		base_idx = index // self.num_variants
		variant_idx = index % self.num_variants
		apply_xflip, apply_yflip = self.variants[variant_idx]

		fname = self._file_list[base_idx]
		data = self._load_file(fname)

		image = data['image']
		condition = data['conditions']

		if apply_xflip:
			image = torch.flip(image, dims=[2])
			condition = torch.flip(condition, dims=[2])
		if apply_yflip:
			image = torch.flip(image, dims=[1])
			condition = torch.flip(condition, dims=[1])

		return image, condition

	def _check_shapes(self, max_mismatches=5):
		print("🔍 Validating image and condition shapes...")
		mismatch_count = 0
		for fname in self._file_list:
			try:
				data = self._load_file(fname)
				img_shape = tuple(data['image'].shape)
				cond_shape = tuple(data['conditions'].shape)

				if img_shape != self.image_shape or cond_shape != self.condition_shape:
					mismatch_count += 1
					print(f"❌ Shape mismatch in {fname}: image {img_shape}, condition {cond_shape}")
					if mismatch_count >= max_mismatches:
						print("⚠ Too many mismatches — stopping early.")
						break
			except Exception as e:
				print(f"⚠ Error reading {fname}: {e}")
				mismatch_count += 1
				if mismatch_count >= max_mismatches:
					break

		if mismatch_count == 0:
			print("✅ All files match the expected shape.")
		else:
			raise ValueError(f"{mismatch_count} mismatched or unreadable samples found.")

	def close(self):
		if self._zipfile is not None:
			self._zipfile.close()
