from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
from spinal_surgery.lab.sensors.ultrasound.simulate_US_conv import USSimulatorConv
import cv2
import numpy as np
import torch


class USSlicer(LabelImgSlicer):
    # Class: US slicer
    # Function: __init__
    # Function: slice_US
    # Function: update_plotter
    def __init__(self, us_cfg, label_maps, ct_maps, if_use_ct, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res=0.0015, max_distance=0.02, # [m]
                 body_label=120, height = 0.13, height_img = 0.132,
                 visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        label maps: list of label maps (3D volumes)
        num_envs: number of environments
        plane_axes: dict of plane axes for imaging, in our case is 'x' and 'z' axes of the ee frame
        x_z_range: range of x and z in mm in the human frame as a rectangle [[min_x, min_z, min_x_angle], [max_x, max_z, max_x_angle]]
        init_x_z_y_angle: initial x, z human position, angle between ee x axis and human x axis in the xz plane 
        of human frame [x, z
        img_size: size of the image [w, h]
        img_res: resolution of the image
        label_res: resolution of the label map
        max_distance: maximum distance for displaying us image
        body_label: label of the body trunc
        height: height of ee above the surface
        height_img: height of the us image frame
        visualize: whether to visualize the human frame
        '''
        super().__init__(label_maps, ct_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, label_res, max_distance,
                 body_label, height, height_img,
                 visualize, plane_axes)
        self.us_sim = USSimulatorConv(us_cfg, device=device)
        self.us_cfg = us_cfg
        self.if_use_ct = if_use_ct

        # construct random maps
        self.construct_T_maps()
        self.construct_Vl_maps()

        # construct images
        self.T0_img_tensor = torch.zeros((self.num_envs, self.img_size[0], self.img_size[1]), 
                                            dtype=torch.float32, 
                                            device=self.device)
        self.T1_img_tensor = torch.zeros((self.num_envs, self.img_size[0], self.img_size[1]),
                                            dtype=torch.float32,
                                            device=self.device)
        self.T0_T1_img_tensor = torch.zeros((self.num_envs, self.img_size[0], self.img_size[1], 2), 
                                            dtype=torch.float32, 
                                            device=self.device)
        self.Vl_img_tensor = torch.zeros((self.num_envs, self.us_cfg['large_scale_resolution'], self.us_cfg['large_scale_resolution']),
                                            dtype=torch.float32,
                                            device=self.device)

    def construct_T_maps(self):
        # create tensors storing the random maps: 
        # parameters: T0, T1, Vl 
        # T0, T1: res: img_res, Vl: res: label_res
        # init position of random map
        # create position list
        # env to human
        self.env_to_human_inds = torch.arange(self.num_envs, device=self.device) % self.n_human_types # (N,)
        self.env_to_human_inds_aug = self.env_to_human_inds.reshape((-1, 1)).repeat(1, self.img_size[0]*self.img_size[1]) # (N, H, W)
        

        self.T_rand_frame_poses = []
        x_z_range_tensor = torch.tensor(self.x_z_range, device=self.device)
        xz_middle = x_z_range_tensor.mean(dim=0)[0:2]
        for i in range(self.n_human_types):
            T_rand_frame_pos = torch.zeros((3,), device=self.device)
            y_mid = (self.surface_map_list[i][xz_middle[0].int(), xz_middle[1].int()].item())
            T_rand_frame_pos[[0, 2]] = x_z_range_tensor[0, 0:2] * self.label_res
            T_rand_frame_pos[1] = y_mid * self.label_res
            T_rand_frame_pos -= max(self.img_real_size) * 0.5
            self.T_rand_frame_poses.append(T_rand_frame_pos)

        self.T_rand_frame_poses = torch.stack(self.T_rand_frame_poses, dim=0) # (n, 3)
        self.T_rand_frame_poses_aug = self.T_rand_frame_poses[self.env_to_human_inds, :] # (num_envs, 3)
        self.T_rand_frame_poses_aug = self.T_rand_frame_poses_aug.reshape((-1, 1, 3))

        size_xz = (x_z_range_tensor[1, 0:2] - x_z_range_tensor[0, 0:2]) * self.label_res / self.img_res
        size_xz += max(self.img_size)
        size_xz = size_xz.int()
        size_y = 2 * max(self.img_size)

        # construct maps
        self.T0_map_mu = torch.zeros((self.n_human_types, size_xz[0], size_y, size_xz[1]), device=self.device)
        self.T0_map_s = torch.ones((self.n_human_types, size_xz[0], size_y, size_xz[1]), device=self.device)
        self.T0_map = torch.normal(self.T0_map_mu, self.T0_map_s)
        self.T1_map = torch.normal(self.T0_map_mu, self.T0_map_s)
        self.T0_T1_map = torch.randn((self.n_human_types, size_xz[0], size_y, size_xz[1], 2), device=self.device) # (n, X, Y, Z, 2)

        self.rand_coords_min = torch.zeros((self.num_envs, self.img_size[0] * self.img_size[1], 3), device=self.device)
        self.rand_coords_max = torch.tensor(self.T0_T1_map[0].shape[:-1], device=self.device).repeat(
                self.num_envs, self.img_size[0] * self.img_size[1], 1) - 1
        


    def construct_Vl_maps(self):
        # construct Vl img coordinates
        l_img_size = self.us_cfg['large_scale_resolution']
        self.env_to_human_inds_Vl = self.env_to_human_inds.reshape((-1, 1)).repeat(1, l_img_size*l_img_size)

        l_arange = torch.arange(l_img_size, device=self.device)
        self.l_x_grid, self.l_z_grid = torch.meshgrid(l_arange - l_img_size//2, 
                                                  l_arange)
        self.l_y_grid = torch.zeros_like(self.l_x_grid, device=self.device)
        self.l_img_coords = torch.stack([self.l_x_grid, self.l_y_grid, self.l_z_grid], dim=-1).reshape((-1, 3)).float() * self.label_res # (l*l, 3)

        # create coordinate system
        self.Vl_rand_frame_poses = []
        x_z_range_tensor = torch.tensor(self.x_z_range, device=self.device)
        xz_middle = x_z_range_tensor.mean(dim=0)[0:2]
        for i in range(self.n_human_types):
            Vl_rand_frame_pos = torch.zeros((3,), device=self.device)
            y_mid = (self.surface_map_list[i][xz_middle[0].int(), xz_middle[1].int()].item())
            Vl_rand_frame_pos[[0, 2]] = x_z_range_tensor[0, 0:2] * self.label_res
            Vl_rand_frame_pos[1] = y_mid * self.label_res
            Vl_rand_frame_pos -= l_img_size * self.label_res
            self.Vl_rand_frame_poses.append(Vl_rand_frame_pos)

        self.Vl_rand_frame_poses = torch.stack(self.Vl_rand_frame_poses, dim=0) # (n, 3)
        self.Vl_rand_frame_poses_aug = self.Vl_rand_frame_poses[self.env_to_human_inds, :] # (num_envs, 3)
        self.Vl_rand_frame_poses_aug = self.Vl_rand_frame_poses_aug.reshape((-1, 1, 3)) # (num_envs, 1, 3)

        # determin 
        l_size_xyz = torch.zeros((3,), device=self.device)
        l_size_xyz[[0, 2]] = (x_z_range_tensor[1, 0:2] - x_z_range_tensor[0, 0:2])
        l_size_xyz[1] = l_img_size
        l_size_xyz = (l_size_xyz + 2*l_img_size).int()

        self.Vl_mu = torch.zeros((self.n_human_types, l_size_xyz[0], l_size_xyz[1], l_size_xyz[2]), device=self.device)
        self.Vl_s = torch.ones((self.n_human_types, l_size_xyz[0], l_size_xyz[1], l_size_xyz[2]), device=self.device)
        self.Vl_map = torch.normal(self.Vl_mu, self.Vl_s) # (n, l, l, l, 2)

        self.Vl_coords_min = torch.zeros((self.num_envs, self.l_img_coords.shape[0], 3), device=self.device)
        self.Vl_coords_max = torch.tensor(self.Vl_map[0].shape, device=self.device).repeat(
                self.num_envs, self.l_img_coords.shape[0], 1) - 1


    def slice_rand_maps(self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        # slice random maps
        # after computing the human img coords
        l_img_size = self.us_cfg['large_scale_resolution']
        self.human_img_l_coords = self.get_human_img_coords(
            self.l_img_coords,
            world_to_human_pos,
            world_to_human_quat,
            world_to_ee_pos,
            world_to_ee_quat) # (num_envs, l*l, 3)
        
        self.rand_frame_img_coords = self.human_img_coords * self.label_res - self.T_rand_frame_poses_aug
        self.rand_frame_img_coords = self.rand_frame_img_coords / self.img_res
        self.rand_frame_img_coords = torch.clamp(
            self.rand_frame_img_coords,
            self.rand_coords_min,
            max=self.rand_coords_max)
        
        self.Vl_frame_img_coords = self.human_img_l_coords * self.label_res - self.Vl_rand_frame_poses_aug
        self.Vl_frame_img_coords = self.Vl_frame_img_coords / self.label_res
        self.Vl_frame_img_coords = torch.clamp(
            self.Vl_frame_img_coords,
            self.Vl_coords_min,
            max=self.Vl_coords_max)
        
        self.T0_T1_img_tensor = self.T0_T1_map[self.env_to_human_inds_aug,
                self.rand_frame_img_coords[:, :, 0].int(), 
                self.rand_frame_img_coords[:, :, 1].int(), 
                self.rand_frame_img_coords[:, :, 2].int(), :
            ].reshape((-1, self.img_size[0], self.img_size[1], 2))
        
        self.Vl_img_tensor = self.Vl_map[self.env_to_human_inds_Vl,
                self.Vl_frame_img_coords[:, :, 0].int(), 
                self.Vl_frame_img_coords[:, :, 1].int(), 
                self.Vl_frame_img_coords[:, :, 2].int()
            ].reshape((-1, l_img_size, l_img_size))


    def slice_US(self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        self.slice_label_img(world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat)
        # self.us_img_tensor = self.us_sim.simulate_US_image(self.label_img_tensor.permute(0, 2, 1), False) # (n, H, W)
        self.slice_rand_maps(world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat)
        self.us_img_tensor = self.us_sim.simulate_US_image_given_rand_map(
            self.label_img_tensor.permute(0, 2, 1), 
            self.T0_T1_img_tensor[:, :, :, 0].permute(0, 2, 1), 
            self.T0_T1_img_tensor[:, :, :, 1].permute(0, 2, 1), 
            self.Vl_img_tensor.permute(0, 2, 1), 
            False,
            self.if_use_ct,
            self.ct_img_tensor.permute(0, 2, 1)) # (n, H, W)


    def visualize(self, first_n=20):
        super().visualize(first_n)
        first_n = min(first_n, self.num_envs)

        combined_US_img = self.us_img_tensor[:first_n, :, :].permute(0, 2, 1).reshape((first_n * self.img_size[0], self.img_size[1])) # (w * first_n, h)

        combined_img_np = combined_US_img.cpu().numpy()

        cv2.imshow("US Image Update", combined_img_np.T / 20)
        cv2.waitKey(1)


