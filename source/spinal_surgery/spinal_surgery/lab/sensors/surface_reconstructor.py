from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
from spinal_surgery.lab.sensors.ultrasound.label_img_slicer import LabelImgSlicer
import torch
import pyvista as pv
from isaaclab.utils.math import quat_from_matrix, combine_frame_transforms, transform_points, subtract_frame_transforms, matrix_from_quat


class SurfaceReconstructor(LabelImgSlicer):
    '''
    Perform low resolution surface reconstruction 
    function:
    - obtain label image slices (with low resolution)
    - obtain ground truth segmentation of bone surface from label images
    - update reconstruction tensor from the label images (in the human frame)
    - update reconstruction tensor in the US frame (as observation)
    - goal: reconstruct a single vertebra
    
    '''
    def __init__(self, label_maps, ct_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 volume_size, volume_res, target_vertebra_center, target_vertebra_label, missing_rate, max_roll_adj,# size of reconstruction volumes
                 add_height, target_vertebra_points,
                 img_size, img_res, img_thickness=1, roll_adj=0.0, label_res=0.0015, max_distance=0.02, # [m]
                 body_label=120, height = 0.13, height_img = 0.13,
                 visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        label maps: list of label maps (3D volumes)
        human_list: list of human types
        num_envs: number of environments
        
        volume_size: size of saved recontruction volume ()
        target_vertebra_center: center of target vertebra (to determine postion of volume in human frame)
        target_vertebra_label: label of target vertebra in segmentation map
        missing_rate: rate of missing data in the reconstruction
        max_roll_adj: maximum roll adjustment for the ee frame
        add_height: addtional height to start to construct the volume

        
        img_size: size of the image, need to be small
        img_res: resolution of the image
        label_res: resolution of the label map
        max_distance: maximum distance for displaying us image
        plane_axes: dict of plane axes for imaging, in our case is 'x' and 'z' axes of the ee frame
        x_z_range: range of x and z in mm in the human frame as a rectangle [[min_x, min_z, min_x_angle], [max_x, max_z, max_x_angle]]
        init_x_z_y_angle: initial x, z human position, angle between ee x axis and human x axis in the xz plane 
        of human frame [x, z
        body_label: label of the body trunc
        height: height of ee above the surface
        height_img: height of the us image frame
        visualize: whether to visualize the human frame
        ''' 
        super().__init__(label_maps, ct_maps, human_list, num_envs, x_z_range, init_x_z_x_angle, device, label_convert_map,
                 img_size, img_res, img_thickness, roll_adj, label_res, max_distance,
                 body_label, height, height_img,
                 visualize, plane_axes)
        
        # construct volumes
        self.volume_size = torch.tensor(volume_size).to(self.device)
        self.human_rec_volume = torch.zeros((self.num_envs,) + tuple(volume_size)).to(self.device) # (num_envs, xv, yv, zv)
        self.add_height = add_height
        
        self.volume_res = volume_res
        self.real_volume_size = torch.tensor(volume_res).to(self.device) * self.volume_size
        self.target_vertebra_center = target_vertebra_center # (N, 3) center of target vertebrae
        self.target_vertebra_label = target_vertebra_label
        self.target_vertebra_points = target_vertebra_points # (n, P, 3) points of target vertebrae
        
        # origin of the human rec volume in the human frame
        self.human_rec_volume_corner = target_vertebra_center - self.real_volume_size / 2
        
        # environment labels for each pixel in the image
        self.img_coods_env_label = torch.arange(self.num_envs).to(self.device).reshape((-1, 1)).repeat(
            1, self.img_size[0] * self.img_thickness)
        
        # missing rate of reconstruction
        self.missing_rate = missing_rate
        
         # volume in ultrasound frame
        self.US_rec_volume = torch.zeros((self.num_envs,) + tuple(volume_size)).to(self.device)
        # point positions of us rec volume
        x_grid, y_grid, z_grid = torch.meshgrid(torch.arange(volume_size[0], device=self.device) - volume_size[0]//2, 
                                                torch.arange(volume_size[1], device=self.device) - volume_size[1]//2,
                                                torch.arange(volume_size[2], device=self.device)) # (xv, yv, zv)
        self.rec_volume_coords = torch.stack([x_grid, y_grid, z_grid], dim=-1).reshape((-1, 3)).float() * volume_res # (xv * yv * zv, 3)
        self.volume_coords_env_labels = torch.arange(self.num_envs).to(self.device).reshape((-1, 1)).repeat(
            1, self.rec_volume_coords.shape[0]) # (N, xv * yv * zv)

        self.max_roll_adj = max_roll_adj * torch.ones((num_envs, 1), device=device)

        # for ray casting:
        x_arange = torch.arange(self.img_size[0], device=self.device)
        z_arange = torch.arange(self.img_thickness, device=self.device)
        self.x_grid, self.z_grid = torch.meshgrid(x_arange, z_arange) # (w, e)
        self.x_grid = self.x_grid.unsqueeze(0).repeat(self.num_envs, 1, 1) # (n, w, e)
        self.z_grid = self.z_grid.unsqueeze(0).repeat(self.num_envs, 1, 1) # (n, w, e)

        # get upper surface
        self.get_target_bone_upper_surface()


        self.vis_rec = visualize
        if visualize:
            first_vertebra = target_vertebra_points[0]
            self.first_vertebra = first_vertebra * self.label_res
            first_vertebra = self.first_vertebra - self.human_rec_volume_corner[0, :]
            first_vertebra = first_vertebra / self.volume_res
            first_vertebra_pv = pv.PolyData(first_vertebra.cpu().numpy())
            self.p_human_rec = pv.Plotter()
            self.p_us_rec = pv.Plotter()
            self.p_human_rec.add_mesh(first_vertebra_pv, color='blue', point_size=0.5)
            self.us_ver_actor = self.p_us_rec.add_mesh(first_vertebra_pv, color='red', point_size=0.5)

            # visualize_reconstructed and complete volume together
            not_coverged = torch.logical_and(self.human_rec_volume[0, :, :, :] == 0, 
                                             self.upper_surface_volume_list[0] == 1).cpu().numpy()
            visualize_volume = self.human_rec_volume[0, :, :, :].cpu().numpy()*200
            visualize_volume[not_coverged] = 100
            self.human_actor=self.p_human_rec.add_volume(visualize_volume, opacity=[0.01, 0.5])

            self.us_actor=self.p_us_rec.add_volume(self.US_rec_volume[0, :, :, :].cpu().numpy()*200, opacity=[0.01, 0.5])
            self.p_human_rec.show_axes()
            self.p_us_rec.show_axes()
            self.p_human_rec.show(interactive_update=True)
            self.p_us_rec.show(interactive_update=True)
            


    def get_volume_to_ee_coords(self, volume_coords, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat):
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat) # (num_envs, 3), (num_envs, 4)
        human_to_ee_rot = matrix_from_quat(human_to_ee_quat) # (num_envs, 3, 3)
        normal_drcts = human_to_ee_rot[:, :, 2]
        prods = normal_drcts @ torch.tensor([0.0, 1.0, 0.0], device=self.device)
        normal_drcts[prods < 0, :] = -normal_drcts[prods < 0, :]
        human_to_ee_pos = (human_to_ee_pos + (self.height_img + self.add_height) * normal_drcts) # (num_envs, 3)
        volume_to_ee_pos = human_to_ee_pos - self.human_rec_volume_corner
        volume_to_ee_coords = transform_points(volume_coords, volume_to_ee_pos, human_to_ee_quat) # (num_envs, xv * yv * zv, 3)
        volume_to_ee_coords = volume_to_ee_coords.reshape((-1, volume_coords.shape[0], volume_coords.shape[1])) / self.volume_res # convert to pixel coords
        # clamp the coords
        volume_to_ee_coords = torch.clamp(
            volume_to_ee_coords, 
            torch.zeros_like(volume_to_ee_coords, device=self.device), 
            max=self.volume_size.reshape((1, 1, -1)).repeat(volume_to_ee_coords.shape[0], volume_to_ee_coords.shape[1], 1) - 1
        )
        return volume_to_ee_coords
    
    def get_target_bone_upper_surface(self):
        '''
        get the target bone upper surface
        '''
        self.upper_surface_volume_list = []
        self.upper_surface_num_points = []
        for i in range(self.n_human_types):
            upper_surface_volume = torch.zeros(tuple(self.volume_size)).to(self.device)
            human_vertebra_points = torch.tensor(self.target_vertebra_points[i]).to(self.device)  # (P, 3)

            volume_vertbebra_points = human_vertebra_points * self.label_res - self.human_rec_volume_corner[i, :]  # (P, 3)
            volume_vertbebra_points = volume_vertbebra_points / self.volume_res  # (P, 3)
            volume_vertbebra_points = volume_vertbebra_points.to(torch.int)
            volume_vertbebra_points = torch.clamp(
                volume_vertbebra_points, 
                torch.zeros_like(volume_vertbebra_points, device=self.device), 
                max=self.volume_size.reshape((1, -1)).repeat(volume_vertbebra_points.shape[0], 1) - 1
            )
            volume_vertbebra_points = volume_vertbebra_points[volume_vertbebra_points[:, 1] < self.volume_size[2] // 2, :] # (P, 3)

            upper_surface_volume[volume_vertbebra_points[:, 0],
                                 volume_vertbebra_points[:, 1],
                                 volume_vertbebra_points[:, 2]] = 1
            self.upper_surface_volume_list.append(upper_surface_volume)
            self.upper_surface_num_points.append(torch.sum(upper_surface_volume == 1))

        self.upper_surface_num_points = torch.tensor(self.upper_surface_num_points).to(self.device) # (N,)

        
        
    def edge_from_label_slice(self, world_to_human_pos, world_to_human_rot, world_to_ee_pos, world_to_ee_quat):
        '''
        obtain position of edges in the human rec volume'''
        # obtain label slices
        self.slice_label_img(world_to_human_pos, world_to_human_rot, world_to_ee_pos, world_to_ee_quat) # (N, W, H, E)
        
        # obtain edge from label slices # bone label: 26-43
        # separately process each human
        bone_map = torch.logical_and(self.label_img_tensor > 25, self.label_img_tensor < 44) # (N, W, H, E)
        bone_map_below = torch.logical_and(self.label_img_tensor[:, :, :-1, :] > 25, self.label_img_tensor[:, :, :-1, :] < 44) # (N, W, H-1, E)
        bone_map_below = torch.cat([torch.zeros_like(bone_map_below[:, :, 0:1, :]).to(self.device), bone_map_below], dim=2)
        bone_map_edge = torch.logical_and(bone_map, torch.logical_not(bone_map_below)) # (N, W, H, E)
        rand_select = torch.rand_like(self.label_img_tensor.to(torch.float)) > self.missing_rate
        
        bone_map_edge = torch.logical_and(bone_map_edge, rand_select)
        bone_map_target = torch.logical_and(self.label_img_tensor == self.target_vertebra_label, bone_map_edge)
        
        return bone_map_edge, bone_map_target
    
    def ray_casting(self, bone_map_edge: torch.Tensor):
        '''
        Only keep the voxels with lowest y value
        '''
        bone_map_edge_y = torch.argmax(bone_map_edge.float(), dim=2) # (N, W*E)
        # if has pixel:
        bone_map_edge_any_y = torch.any(bone_map_edge, dim=2) # (N, W*E)
        edge_coords = torch.cat([
            self.x_grid[bone_map_edge_any_y].unsqueeze(-1),
            bone_map_edge_y[bone_map_edge_any_y].unsqueeze(-1),
            self.z_grid[bone_map_edge_any_y].unsqueeze(-1),
        ], dim=-1).reshape((-1, 3)) # (K_N, 3)

        # env labels
        env_labels = self.img_coods_env_label[bone_map_edge_any_y.view(self.num_envs, -1)].unsqueeze(1) # (K_N, 1)

        human_edge_coords = self.human_img_coords[
            env_labels[:, 0],
            self.img_size[1] * edge_coords[:, 0].to(torch.int) + edge_coords[:, 1].to(torch.int),
            :,
        ] # (K_N, 3)

        return human_edge_coords, env_labels
    
    
    def get_human_to_edge_pos(self, bone_map_edge):
        '''
        human_img_coords: (N, w*h*e, 3)
        bone_map_edge: (N, W, H, E)
        '''
        # (num_envs, w*h*e, 3)
        human_edge_coords, env_labels = self.ray_casting(bone_map_edge) # (K_N, 3)
        human_edge_pos = human_edge_coords * self.label_res # (K_N, 3)
        
        
        return env_labels, human_edge_pos
        
    def update_human_rec(self,  env_labels, human_edge_pos):
        '''
        human_edge_pos: (K_N, 3)
        env_labels: (K_N, 1)
        '''
        # convert to human rec volume coords
        human_edge_pos = human_edge_pos - self.human_rec_volume_corner[env_labels[:, 0], :] # (K_N, 3)
        human_edge_coords = (human_edge_pos / self.volume_res) # (K_N, 3)

        human_edge_coords = torch.clamp(
            human_edge_coords,
            torch.ones_like(human_edge_coords, device=self.device),
            max=self.volume_size.reshape((1, -1)).repeat(human_edge_coords.shape[0], 1) - 2 
        ) # leave 0 for the outer boundary, then US rec volume will have no weird padding

        # only take above half
        above_half = human_edge_coords[:, 1] < self.volume_size[2] // 2
        human_edge_coords = human_edge_coords[above_half, :] # (K_N, 3)
        env_labels = env_labels[above_half, :] # (K_N, 1)
        
        if human_edge_pos.shape[1]== 0:
            return
        self.human_rec_volume[env_labels[:, 0], 
                              human_edge_coords[:, 0].to(torch.int), 
                              human_edge_coords[:, 1].to(torch.int), 
                              human_edge_coords[:, 2].to(torch.int)] = 1
        
        # compute coverage
        # get target coverage
        cur_cov = torch.sum(self.human_rec_volume, dim=(1, 2, 3))
        self.incremental_cov = cur_cov - self.cur_cov
        self.cur_cov = cur_cov
        
        
    def update_US_rec(self, world_to_human_pos, world_to_human_rot, world_to_ee_pos, world_to_ee_quat):
        self.US_rec_volume[:, :, :, :] = 0
        volume_to_ee_coords = self.get_volume_to_ee_coords(
            self.rec_volume_coords, 
            world_to_human_pos, 
            world_to_human_rot, 
            world_to_ee_pos, world_to_ee_quat) # (num_envs, xv * yv * zv, 3)
        
        self.US_rec_volume[:, :, :, :] = self.human_rec_volume[self.volume_coords_env_labels.int(), 
            volume_to_ee_coords[:, :, 0].int(), 
            volume_to_ee_coords[:, :, 1].int(), 
            volume_to_ee_coords[:, :, 2].int()
        ].reshape(self.US_rec_volume.shape) # (num_envs, xv, yv, zv)
        
    
    def obtain_new_reconstruction_from_ee_pose(self, world_to_human_pos, world_to_human_rot, world_to_ee_pos, world_to_ee_quat):
        '''
        obtain new reconstruction from ee pose
        '''
        # get edge from label slice
        bone_map_edge, bone_map_target = self.edge_from_label_slice(
            world_to_human_pos, world_to_human_rot, 
            world_to_ee_pos, world_to_ee_quat)
        
        # get human edge positions in human frame
        env_labels, human_edge_pos = self.get_human_to_edge_pos(bone_map_target) # (K_N, 3)

        # update reconstructions
        self.update_human_rec(env_labels, human_edge_pos)
        self.update_US_rec(world_to_human_pos, world_to_human_rot, world_to_ee_pos, world_to_ee_quat)

        if self.vis_rec:
            ee_to_human_pos, ee_to_human_quat = subtract_frame_transforms(
                world_to_ee_pos, world_to_ee_quat, world_to_human_pos, world_to_human_rot)
            ver_in_us = transform_points(self.first_vertebra, ee_to_human_pos, ee_to_human_quat) # (N, 3)
            ver_in_us[:, :, 2] = ver_in_us[:, :, 2] - self.height_img - self.add_height# (N, 3)
            ver_in_us[:, :, 0] += self.real_volume_size[0] / 2
            ver_in_us[:, :, 1] += self.real_volume_size[1] / 2
            ver_in_us = ver_in_us / self.volume_res
            ver_actor = self.p_us_rec.add_mesh(pv.PolyData(ver_in_us[0, :, :].cpu().numpy()), color='red', point_size=0.5)
            self.p_us_rec.remove_actor(self.us_ver_actor)
            self.us_ver_actor=ver_actor
            # update human rec volume

            not_coverged = torch.logical_and(self.human_rec_volume[0, :, :, :] == 0, 
                                             self.upper_surface_volume_list[0] == 1).cpu().numpy()
            visualize_volume = self.human_rec_volume[0, :, :, :].cpu().numpy()*200
            visualize_volume[not_coverged] = 100
            human_actor=self.p_human_rec.add_volume(visualize_volume, opacity=[0.01, 0.5])
            # human_actor=self.p_human_rec.add_volume(self.human_rec_volume[0, :, :, :].cpu().numpy()*200, opacity=[0.01, 0.5])
            us_actor=self.p_us_rec.add_volume(self.US_rec_volume[0, :, :, :].cpu().numpy()*200, cmap=['cyan', 'green'], opacity=[0.01, 0.5])
            self.p_human_rec.remove_actor(self.human_actor)
            self.p_us_rec.remove_actor(self.us_actor)
            self.human_actor=human_actor
            self.us_actor=us_actor
            self.p_human_rec.update()
            self.p_us_rec.update()


    def get_converage_ratio(self):
        # get total number of points for surface:
        upper_surface_per_env = self.upper_surface_num_points[self.env_to_human_inds]

        return self.cur_cov / upper_surface_per_env  # (N,)

    def update_cmd(self, d_x_z_z_y_angle):
        # update x, z, position and y angle
        super().update_cmd(d_x_z_z_y_angle[:, 0:3])
        # update  roll_adujst
        self.roll_adj += d_x_z_z_y_angle[:, 3:4]
        self.roll_adj = torch.clamp(self.roll_adj, min=-self.max_roll_adj, max=self.max_roll_adj)

    def human_cmd_state_from_ee_pose(self, human_to_target_pos, human_to_target_quat):
        '''
        convert ee_pose to human command'''
        
        rot_mat = matrix_from_quat(human_to_target_quat)
        z_axis = rot_mat[:, :, 2]  # (num_envs, 3)
        x_axis = rot_mat[:, :, 0]  # (num_envs, 3)
        angle_y = torch.atan2(x_axis[:, 2], x_axis[:, 0])  # (num_envs)

        pos = human_to_target_pos + z_axis * self.height
        x_z = pos[:, [0, 2]] / self.label_res

        # compute real roll adjustment
        y_axis = rot_mat[:, :, 1]  # (num_envs, 3)
        suface_normal = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.n_human_types):
            cur_x = x_z[i::self.n_human_types, 0]  # (num_envs / n, 3)
            cur_z = x_z[i::self.n_human_types, 1]  # (num_envs / n, 3)
            cur_x = torch.clamp(cur_x, min=0, max=self.label_maps[i].shape[0] - 1)
            cur_z = torch.clamp(cur_z, min=0, max=self.label_maps[i].shape[2] - 1)
            
            normal = self.surface_normal_list[i][cur_x.int(), cur_z.int(), :]  # (num_envs / n, 3)
            suface_normal[i::self.n_human_types, :] = normal
        sin_angle = torch.sum(torch.cross(suface_normal, y_axis) * z_axis, dim=-1)  # cross product and dot
        angle_z = torch.asin(sin_angle)  # (num_envs)

        return torch.cat([x_z, angle_y.unsqueeze(-1), angle_z.unsqueeze(-1)], dim=-1)  # (num_envs, 3)

    def reset(self):
        volume_size = self.volume_size.int().cpu().numpy()
        self.human_rec_volume = torch.zeros((self.num_envs,) + tuple(volume_size)).to(self.device)
        self.US_rec_volume = torch.zeros((self.num_envs,) + tuple(volume_size)).to(self.device)
        self.roll_adj = torch.zeros((self.num_envs, 1), device=self.device)
        self.incremental_cov = torch.zeros((self.num_envs,), device=self.device)
        self.cur_cov = torch.zeros((self.num_envs,), device=self.device)


        
        
        
        
