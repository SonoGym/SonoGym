import torch
import torch.nn.functional as F
from monai.networks.nets.unet import UNet
import os
from PIL import Image
import torchvision.transforms as transforms
from spinal_surgery import PACKAGE_DIR, PROJECT_DIR
import numpy as np


class USSimulatorNetwork:
    def __init__(self, us_model_cfg, device):
        self.device = device
        model_paths = us_model_cfg["model_path"]
        self.model_list = []
        for model_path in model_paths:

            model_path = os.path.join(PROJECT_DIR, model_path)

            self.model = UNet(
                spatial_dims=us_model_cfg["model"]["spatial_dims"],
                in_channels=us_model_cfg["model"]["in_channels"],
                out_channels=us_model_cfg["model"]["out_channels"],
                channels=us_model_cfg["model"]["channels"],
                strides=us_model_cfg["model"]["strides"],
                num_res_units=us_model_cfg["model"]["num_res_units"],
                dropout=us_model_cfg["model"]["dropout"],
                act=("leakyrelu", {"negative_slope": 0.2}),
            ).to(self.device)
            # get model
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            self.model_list.append(self.model)

        # get CT cfg
        self.CT_cfg = us_model_cfg["CT"]

        # label res
        self.label_res = us_model_cfg["label_res"]
        self.image_size = self.CT_cfg["size"]

        self.cfg = us_model_cfg

        self.construct_train_data_histogram()

        # change the hist every k steps
        self.k = us_model_cfg["reset_hist_interval"]
        self.model_k = us_model_cfg["model_change_interval"]
        self.step = 0

    def simulate_US_image(self, ct_img_tensor):
        """
        simulate US image from CT image
        ct_img_tensor: (num_envs*e, 1, w, h)
        """

        with torch.no_grad():

            # intensity scale: [-1, 1]
            # print('ct', torch.min(ct_img_tensor), torch.max(ct_img_tensor))
            ct = torch.clamp(
                ct_img_tensor, self.CT_cfg["range"][0], self.CT_cfg["range"][1]
            )
            ct = (ct - self.CT_cfg["range"][0]) / (
                self.CT_cfg["range"][1] - self.CT_cfg["range"][0]
            )  # * 2 - 1

            # spatial resolution
            ct = F.interpolate(
                ct,
                size=(self.image_size[0], self.image_size[1]),
                mode="nearest-exact",
                # align_corners=False,
            )

            # down sample and upsample it
            ct = F.interpolate(
                ct,
                size=(self.image_size[0] // 4, self.image_size[1] // 4),
                mode="nearest-exact",
                # align_corners=False,
            )
            ct = F.interpolate(
                ct,
                size=(self.image_size[0], self.image_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            ct = self.test_match_train(ct)

            # simulate US image
            us = self.model(ct)

            # change back
            us_img_tensor = F.interpolate(
                us,
                size=(ct_img_tensor.shape[-2], ct_img_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            # ct_img_tensor = F.interpolate(
            #     ct,
            #     size=(ct_img_tensor.shape[-2], ct_img_tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )

        return us_img_tensor.detach()

    def read_img_folder(self, folder_path):
        """
        Read all images in a folder and return them as a list of tensors.
        """
        # Define the transformation to convert images to tensors
        transform = transforms.ToTensor()

        # List to hold the image tensors
        image_tensors = []

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png") or filename.endswith(".jpeg"):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("L")
                img_tensor = transform(img)

                #  Append the tensor to the list
                image_tensors.append(img_tensor)

        # Stack the list of tensors into a single tensor
        self.train_samples = torch.stack(image_tensors)
        self.train_samples = self.train_samples.to(self.device)

    def construct_train_data_histogram(self):
        source_path = os.path.join(PROJECT_DIR, self.cfg["train_data_sample_path"])
        self.num_bins = self.cfg["num_bins"]
        device = self.device
        self.read_img_folder(source_path)

        # normalize
        source_flat = self.train_samples.flatten()
        self.src_min, self.src_max = source_flat.min(), source_flat.max()
        source_norm = (source_flat - self.src_min) / (
            self.src_max - self.src_min + 1e-8
        )

        self.read_img_folder(source_path)
        # Compute the histogram of the training samples
        # Histogram and CDF
        self.bin_edges = torch.linspace(
            0.0, 1.0, steps=self.num_bins + 1, device=device
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        self.source_hist = torch.histc(
            source_norm, bins=self.num_bins, min=0.0, max=1.0
        )
        self.source_cdf = torch.cumsum(self.source_hist, dim=0)
        self.source_cdf /= self.source_cdf.clone()[-1]

    def test_match_train(self, test: torch.Tensor) -> torch.Tensor:
        """
        Vectorized histogram matching using PyTorch, fully differentiable and GPU-friendly.
        Assumes source and test are 2D torch tensors.
        """

        # Normalize to [0, 1]
        if self.step % self.model_k == 0:
            m = np.random.randint(0, len(self.model_list))
            self.model = self.model_list[m]

        if self.step % self.k == 0:
            self.test_min, self.test_max = test.min(), test.max()

        test_norm = (test - self.test_min) / (self.test_max - self.test_min + 1e-8)

        test_norm = test_norm.flatten()

        if self.step % self.k == 0:
            # Update histogram
            test_hist = torch.histc(test_norm, bins=self.num_bins, min=0.0, max=1.0)
            self.test_cdf = torch.cumsum(test_hist, dim=0)
            self.test_cdf /= self.test_cdf.clone()[-1]

        # Digitize source
        test_bins = torch.bucketize(test_norm, self.bin_edges[:-1], right=False)
        test_bins = torch.clamp(test_bins, 1, self.num_bins) - 1

        # Map: for each source bin, find the closest ref bin with >= cdf
        mapping_indices = torch.searchsorted(self.source_cdf, self.test_cdf)
        mapping_indices = torch.clamp(mapping_indices, 0, self.num_bins - 1)

        # Build LUT: source bin center → matched value
        lut = self.bin_centers[mapping_indices]

        matched_norm = lut[test_bins]
        matched = matched_norm * (self.src_max - self.src_min) + self.src_min

        self.step += 1

        return matched.reshape_as(test)
