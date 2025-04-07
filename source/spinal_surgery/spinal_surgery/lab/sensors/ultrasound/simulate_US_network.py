import torch
import torch.nn.functional as F
from monai.networks.nets.unet import UNet

class USSimulatorNetwork:
    def __init__(self, us_model_cfg, device):
        self.device = device

        self.model = UNet(
            spatial_dims=us_model_cfg['model']['spatial_dims'],
            in_channels=us_model_cfg['model']['in_channels'],
            out_channels=us_model_cfg['model']['out_channels'],
            channels=us_model_cfg['model']['channels'],
            strides=us_model_cfg['model']['strides'],
            num_res_units=us_model_cfg['model']['num_res_units'],
            dropout=us_model_cfg['model']['dropout'],
            act=('leakyrelu', {"negative_slope": 0.2})
        ).to(self.device)
        # get model
        self.model.load_state_dict(torch.load(us_model_cfg['model_path']))
        self.model.eval()

        # get CT cfg
        self.CT_cfg = us_model_cfg["CT"]

        # label res
        self.label_res = us_model_cfg['label_res']
        self.image_size = self.CT_cfg['size']


    def simulate_US_image(self, ct_img_tensor):
        '''
        simulate US image from CT image
        ct_img_tensor: (num_envs*e, 1, w, h)
        '''
        with torch.no_grad():

            # intensity scale: [-1, 1]
            ct = torch.clamp(ct_img_tensor, self.CT_cfg["range"][0], self.CT_cfg["range"][1])
            ct = (ct - self.CT_cfg["range"][0]) / (self.CT_cfg["range"][1] - self.CT_cfg["range"][0]) * 2 - 1

            # spatial resolution
            ct = F.interpolate(
                ct,
                size=(self.image_size[0], self.image_size[1]),
                mode="bilinear",
                align_corners=False,
            )

            # simulate US image
            us = self.model(ct)

            # change back
            us_img_tensor = F.interpolate(
                us,
                size=(ct_img_tensor.shape[-2], ct_img_tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            

        return us_img_tensor.detach()