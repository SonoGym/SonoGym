import numpy as np
import torch
import pyvista as pv
from PIL import Image as im
import time
import matplotlib.pyplot as plt
# from .simulate_US_slice import slice_tensor, load_itk, get_pixel_coordinates
import torch.nn.functional as F
import pydicom
from scipy.ndimage import zoom, gaussian_filter

class USSimulatorConv:
    '''
    The class used to simulate ultrasound imaging, based on an existing label image sliced from the label map
    
    parameters for system:
    f: frequency
    initial energy
    point spread function: sigmax, sigmay
    element size e

    imfusion parameters of the tissue:
    alpha: attenuation
    Z: acoustic impedance
    n: for surface heterogenity
    
    T(x): mu0, mu1, sigma0

    given a position x of pixel a in world coord, with shooting line d and distance l from probe, we have: 
    E(l) = I(l) * cos a^n * ((z1 - z2) / (z1 + z2))^2 * H(x) x G(x)
    R(l) = I(l) * H(x) x T(x)

    by default, we assume n=1
    '''
    def __init__(
        self,
        us_cfg,
        device,
    ) -> None:
        
        system_params=us_cfg["system_params"]
        label_to_params_dict=us_cfg["label_to_ac_params_dict"]
        kernel_size=tuple(us_cfg["kernel_size"])
        E_S_ratio=us_cfg["E_S_ratio"]
        self.device = device
        
        self.f = system_params["frequency"]
        self.I0 = system_params["I0"]
        self.e = system_params["element_size"]
        self.sx_E = system_params["sx_E"]
        self.sy_E = system_params["sy_E"]
        self.sx_B = system_params["sx_B"]
        self.sy_B = system_params["sy_B"]
        self.beta = system_params["TGC_beta"]
        self.beta_edge = system_params["TGC_edge"]

        self.n_I = system_params["noise_I"]
        self.n_mu0 = system_params["noise_mu0"]
        self.n_mu1 = system_params["noise_mu1"]
        self.n_s0 = system_params['noise_s0']
        self.n_f = system_params['noise_f']

        # self.f, self.I0, self.e, self.sx_E, self.sy_E, self.sx_B, self.sy_B, self.beta = system_params
        self.label_to_params_dict = label_to_params_dict
        for key, item in self.label_to_params_dict.items():
            self.label_to_params_dict[key] = torch.tensor([
                item['alpha'], 
                item['z'], 
                item['mu0'], 
                item['mu1'], 
                item['s0'],
                item['Al'],
                item['fl']], device=self.device)
        self.kernel_size = kernel_size

        self.PSF_E = self.compute_PSF_kernel(self.sx_E, self.sy_E)
        self.PSF_B = self.compute_PSF_kernel(self.sx_B, self.sy_B)
        self.E_S_ratio = E_S_ratio

        self.if_large_scale_speckle = us_cfg["large_scale_speckle"]
        self.l_size = us_cfg["large_scale_resolution"]
        if self.if_large_scale_speckle:
            self.large_rand_param_map = torch.zeros((1, 1, 1, 3))

        self.param_map = torch.zeros((1, 1, 1, 3))

        pass

    
    def compute_PSF_kernel(self, sx, sy):
        '''
        get point spread function with kernel size
        '''
        center = (torch.tensor(self.kernel_size, device=self.device)-1) / 2
        print(center)

        x_inds = torch.arange(0, self.kernel_size[0], device=self.device)
        y_inds = torch.arange(0, self.kernel_size[1], device=self.device)
        x_grid, y_grid = torch.meshgrid(x_inds, y_inds)

        x = x_grid - center[0] # (W, H)
        y = y_grid - center[1]
        density_x = torch.exp(-0.5*x**2 / sx**2) * torch.cos(2 * torch.pi * self.f * x)
        density_y = torch.exp(-0.5*y**2 / sy**2)

        PSF_kernel = density_x * density_y
        PSF_kernel = PSF_kernel.reshape((1, 1,) + self.kernel_size)

        return PSF_kernel

        


    def assign_alpha_map(self, label_img: torch.Tensor):
        '''
        assign alpha to each pixel
        '''
        self.alpha_map = torch.zeros(label_img.shape, device=label_img.device)
        for label in self.label_to_params_dict.keys():
            self.alpha_map[label_img==label] = self.label_to_params_dict[label][0] - self.beta

        return self.alpha_map


    def assign_impedance_map(self, label_img: torch.Tensor):
        '''
        assign impedance
        '''
        self.z_map = torch.zeros(label_img.shape, device=label_img.device)
        for label in self.label_to_params_dict.keys():
            self.z_map[label_img==label] = self.label_to_params_dict[label][1]

        return self.z_map
    

    def assign_T_params_map(self, label_img: torch.Tensor):
        '''
        assign mu_0, mu1, sigma0 to T
        '''
        if not label_img.shape==self.rand_param_map.shape[:-1]:
            self.rand_param_map = torch.zeros(label_img.shape + (3,), device=label_img.device)
        if self.if_large_scale_speckle:
            if not label_img.shape==self.large_rand_param_map.shape[:-1]:
                self.large_rand_param_map = torch.zeros(label_img.shape + (2,), device=label_img.device)
        labels = torch.unique(label_img)
        for i in range(labels.shape[0]):
            label = labels[i].item()
            label_items = label_img==label
            self.rand_param_map[label_items, :] = self.label_to_params_dict[label][2:5]
            if self.if_large_scale_speckle:
                self.large_rand_param_map[label_items, :] = self.label_to_params_dict[label][5:7]


        return self.rand_param_map


    def assign_params_map(self, label_img: torch.Tensor):
        '''
        assign all parameters to each pixel
        '''
        if not label_img.shape==self.param_map.shape[:-1]:
            self.param_map = torch.zeros(label_img.shape + (7,), device=label_img.device)
        labels = torch.unique(label_img)
        for i in range(labels.shape[0]):
            label = labels[i].item()
            label_items = label_img==label
            self.param_map[label_items, :] = self.label_to_params_dict[label][:]
            
        return self.param_map
    

    def compute_attenuation_map(self, alpha_map: torch.Tensor):
        '''
        compute attenuation based on alpha
        '''
        alpha_l_map = torch.cumsum(alpha_map, dim=1) * self.e
        atten_map = torch.exp(-alpha_l_map * self.f)
        return atten_map


    def compute_edge_map(self, label_img: torch.Tensor):
        '''
        (n, H, W)
        Compute the edge between labels
        '''

        edge_map = torch.zeros(label_img.shape, device=label_img.device)
        pad_img = F.pad(label_img, (1, 1, 1, 1), 'reflect')

        label_up = pad_img[:, :-2, 1:-1]
        label_down = pad_img[:, 1:-1, 1:-1]

        if_edge = torch.logical_not(label_up==label_down)
        edge_map[if_edge] = 1

        return edge_map
    
    def compute_ct_edge_map(self, ct_img: torch.Tensor):
        '''
        (n, H, W)
        Compute the edge between labels
        '''

        edge_map = torch.zeros(ct_img.shape, device=ct_img.device)
        pad_img = F.pad(ct_img, (1, 1, 1, 1), 'reflect')

        label_up = pad_img[:, :-2, 1:-1]
        label_down = pad_img[:, 1:-1, 1:-1]

        if_edge = abs(label_up-label_down) > (label_down * 0.3)
        edge_map[if_edge] = 1

        return edge_map
    

    def compute_image_gradient(self, img: torch.Tensor):
        '''
        compute gradient of image in order to find surface normal
        Here we mainly focus on finding the surface normal, and we care about 
        angle between x and normal. 
        So here we average the grad in y axis, and only take finite difference of x w.r.t. above
        img: (n, H, W)
        '''
        p1d = (1, 1, 1, 1)
        pad_img = F.pad(img, p1d, 'reflect') 

        img_xm = pad_img[:, :-2, 1:-1]
        img_ym = pad_img[:, 1:-1, :-2]
        img_yp = pad_img[:, 1:-1, 2:]

        grad_x = img - img_xm
        grad_ym = img - img_ym
        grad_yp = img_yp - img
        grad_y = 0.5 * (grad_ym + grad_yp)

        return torch.stack([grad_x, grad_y], dim=-1) # (n, X, Y, 2)
    

    def compute_cos_map(self, img_grad: torch.Tensor):
        cos_map = img_grad[:, :, :, 0] / (torch.linalg.norm(img_grad, dim=-1) + 1e-5)
        return cos_map
    
    
    def generate_noise_map(self, label_img: torch.Tensor):
        '''
        generate random noise that simulate the real US effects
        '''
        # generate random noise
        r0_map = torch.normal(torch.zeros(label_img.shape, device=self.device), torch.ones(label_img.shape, device=self.device))
        r1_map = torch.normal(torch.zeros(label_img.shape, device=self.device), torch.ones(label_img.shape, device=self.device))
        n_map = r0_map * self.n_s0 + self.n_mu0
        n_map_zero = torch.logical_not(r1_map <= self.n_mu1)
        n_map[n_map_zero] = 0

        # get TGC effect
        beta_map = self.beta * torch.ones(label_img.shape, device=self.device)
        beta_l_map = torch.cumsum(beta_map, dim=1) * self.e
        TGC_map = torch.exp(beta_l_map * self.n_f)

        return n_map * TGC_map * self.n_I
    

    def simulate_US_image(self, label_img: torch.Tensor, if_noise=True):
        '''
        img: (n, H. W)
        simulate us image based on label img
        '''
        # initial energy
        self.I0_map = torch.ones(label_img.shape, device=self.device) * self.I0

        # assign parameters
        params_map = self.assign_params_map(label_img)
        alpha_map = params_map[:, :, :, 0]
        z_map = params_map[:, :, :, 1]
        T_params_map = params_map[:, :, :, 2:5]

        # visualize_img(alpha_map[0, :, :].cpu().numpy())
        # visualize_img(z_map[0, :, :].cpu().numpy())
        # visualize_img(T_params_map[0, :, :, 0].cpu().numpy())

        # compute items
        atten_map = self.compute_attenuation_map(alpha_map)
        edge_map = self.compute_edge_map(label_img)

        # visualize_img(atten_map[0, :, :].cpu().numpy())
        # visualize_img(edge_map[0, :, :].cpu().numpy())

        # compute angle map
        edge_grad = self.compute_image_gradient(edge_map)
        cos_map = self.compute_cos_map(edge_grad)
        # cos_map[cos_map<0] = 0

        # visualize_img(cos_map.cpu().numpy())
        # visualize_img(self.PSF_kernel[0, 0, :, :].cpu().numpy())

        # compute z difference
        pad_z_map = F.pad(z_map, (1, 1, 1, 1), mode='reflect')
        z2_map = z_map
        z1_map = pad_z_map[:, :-2, 1:-1]

        # visualize_img((edge_map * cos_map).cpu().numpy())
        # visualize_img(((z1_map - z2_map)**2 / (z1_map + z2_map + 1e-5)**2).cpu().numpy())

        # compute reflection term
        E_map = self.I0_map * atten_map 
        E_map = E_map * edge_map 
        E_map = E_map * (z1_map - z2_map)**2 / (z1_map + z2_map + 1e-5)**2
        E_map *= cos_map
        E_map = E_map[:, None, :, :]
        E_map = F.conv2d(input=E_map, weight=self.PSF_E, stride=1, padding='same')[:, 0, :, :]

        # visualize_img(E_map.cpu().numpy(), True)


        # construct random pattern:
        T0_map = torch.normal(torch.zeros(label_img.shape,device=self.device), torch.ones(label_img.shape,device=self.device))
        T1_map = torch.normal(torch.zeros(label_img.shape, device=self.device), torch.ones(label_img.shape, device=self.device))
        S_map = T0_map * T_params_map[:, :, :, 2] + T_params_map[:, :, :, 0]
        S_map_zero = torch.logical_not(T1_map <= T_params_map[:, :, :, 1])
        S_map[S_map_zero] = 0

        # consider large scale speckle
        if self.if_large_scale_speckle:
            self.large_rand_param_map = params_map[:, :, :, 5:7]
            Vl_map = torch.normal(torch.zeros((label_img.shape[0], self.l_size, self.l_size), device=self.device), 
                                  torch.ones((label_img.shape[0], self.l_size, self.l_size), device=self.device)) # (n, l, l)
            Al_map = self.large_rand_param_map[:, :, :, 0] # (n, H, W)
            fl_map = self.large_rand_param_map[:, :, :, 1] # (n, H, W)
            inds_n, inds_h, inds_w = torch.meshgrid(torch.arange(0, label_img.shape[0], device=self.device), 
                                  torch.arange(0, label_img.shape[1], device=self.device), 
                                  torch.arange(0, label_img.shape[2], device=self.device))
            inds = torch.stack([inds_n, inds_h, inds_w], dim=-1) # (n, H, W, 3)
            inds_lower = (inds / fl_map[:, :, :, None]).long()
            S_map = S_map * (1 + Al_map * Vl_map[inds_lower[:, :, :, 0], inds_lower[:, :, :, 1], inds_lower[:, :, :, 2]])


        S_map = S_map[:, None, :, :]
        B_map = self.I0_map * atten_map * F.conv2d(S_map, self.PSF_B, padding='same')[:, 0, :, :]


        US = self.E_S_ratio * E_map + B_map

        # add noise
        if if_noise:
            noise_map = self.generate_noise_map(label_img=label_img)

            US = US + noise_map

        return US

    def simulate_US_image_given_rand_map(self, 
                                         label_img: torch.Tensor, 
                                         T0_img: torch.Tensor, 
                                         T1_img: torch.Tensor, 
                                         Vl_img: torch.Tensor, 
                                         if_noise=True,
                                         if_ct=False,
                                         ct_img=None):
        '''
        img: (n, H. W)
        simulate us image based on label img
        '''
        # initial energy
        self.I0_map = torch.ones(label_img.shape, device=label_img.device) * self.I0

        # assign parameters
        params_map = self.assign_params_map(label_img)
        alpha_map = params_map[:, :, :, 0]
        if if_ct:
            z_map = ct_img * 1e2
        else:
            z_map = params_map[:, :, :, 1]
        T_params_map = params_map[:, :, :, 2:5]

        # visualize_img(alpha_map[0, :, :].cpu().numpy())
        # visualize_img(z_map[0, :, :].cpu().numpy())
        # visualize_img(T_params_map[0, :, :, 0].cpu().numpy())

        # compute items
        atten_map = self.compute_attenuation_map(alpha_map - self.beta)
        atten_map_E = self.compute_attenuation_map(alpha_map - self.beta_edge)
        if if_ct:
            edge_map = self.compute_ct_edge_map(ct_img)
        else:
            edge_map = self.compute_edge_map(label_img)

        # visualize_img(atten_map[0, :, :].cpu().numpy())
        # visualize_img(edge_map[0, :, :].cpu().numpy())

        # compute angle map
        edge_grad = self.compute_image_gradient(edge_map)
        cos_map = self.compute_cos_map(edge_grad)
        # cos_map[cos_map<0] = 0

        # visualize_img(cos_map.cpu().numpy())
        # visualize_img(self.PSF_kernel[0, 0, :, :].cpu().numpy())

        # compute z difference
        pad_z_map = F.pad(z_map, (1, 1, 1, 1), mode='reflect')
        z2_map = z_map
        z1_map = pad_z_map[:, :-2, 1:-1]

        # visualize_img((edge_map * cos_map).cpu().numpy())
        # visualize_img(((z1_map - z2_map)**2 / (z1_map + z2_map + 1e-5)**2).cpu().numpy())

        # compute reflection term
        E_map = self.I0_map * atten_map_E
        E_map = E_map * (z1_map - z2_map)**2 / (z1_map + z2_map + 1e-5)**2
        if not if_ct:
            E_map = E_map * edge_map 
            E_map *= cos_map
        E_map = E_map[:, None, :, :]
        E_map[E_map>0.1] = 0.1
        E_map = F.conv2d(input=E_map, weight=self.PSF_E, stride=1, padding='same')[:, 0, :, :]

        # visualize_img(E_map.cpu().numpy(), True)


        # construct random pattern:
        T0_map = T0_img # (n, H, W)
        T1_map = T1_img # (n, H, W)
        S_map = T0_map * T_params_map[:, :, :, 2] + T_params_map[:, :, :, 0]
        S_map_zero = torch.logical_not(T1_map <= T_params_map[:, :, :, 1])
        S_map[S_map_zero] = 0

        # consider large scale speckle
        if self.if_large_scale_speckle:
            Vl_map = Vl_img # (n, l, l)
            self.large_rand_param_map = params_map[:, :, :, 5:7]
            Al_map = self.large_rand_param_map[:, :, :, 0] # (n, H, W)
            fl_map = self.large_rand_param_map[:, :, :, 1] # (n, H, W)
            inds_n, inds_h, inds_w = torch.meshgrid(torch.arange(0, label_img.shape[0], device=self.device), 
                                  torch.arange(0, label_img.shape[1], device=self.device), 
                                  torch.arange(0, label_img.shape[2], device=self.device))
            inds = torch.stack([inds_n, inds_h, inds_w], dim=-1) # (n, H, W, 3)
            inds_lower = (inds / fl_map[:, :, :, None]).long()
            S_map = S_map * (1 + Al_map * Vl_map[inds_lower[:, :, :, 0], inds_lower[:, :, :, 1], inds_lower[:, :, :, 2]])


        S_map = S_map[:, None, :, :]
        B_map = self.I0_map * atten_map * F.conv2d(S_map, self.PSF_B, padding='same')[:, 0, :, :]


        US = self.E_S_ratio * E_map + B_map

        # add noise
        if if_noise:
            noise_map = self.generate_noise_map(label_img=label_img)

            US = US + noise_map

        return US


def visualize_img(img, if_gray=False):
    fig, ax = plt.subplots(1, 1)
    if if_gray:
        ax.imshow(
            img, 
            resample=False,
            cmap='gray',
            norm=None
        )
    else:
        ax.imshow(
            img, 
            resample=False,
            norm=None
        )
    plt.plot()
    plt.show()




    







