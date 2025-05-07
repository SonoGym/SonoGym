import torch
import pyvista as pv
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms 
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, transform_points
import numpy as np

class VertebraViewer:
    def __init__(self, num_envs, n_human_types, vertebra_file_list, traj_file_list, if_vis, res, device):
        self.num_envs = num_envs
        self.n_human_types = n_human_types
        self.vertebra_file_list = vertebra_file_list
        self.device = device
        self.if_vis = if_vis
        self.res = res

        # target bone points
        self.vertebra_points_np_list = [pv.read(vertebra_file).points for vertebra_file in vertebra_file_list]
        self.vertebra_points_list = [torch.tensor(self.vertebra_points_np, device=device)
                                           for self.vertebra_points_np in self.vertebra_points_np_list] # (n, P_i, 3)

        # target bone centers
        self.human_to_ver_per_human = [torch.mean(vertebra_points, dim=0) for vertebra_points in self.vertebra_points_list]
        self.human_to_ver_per_human = torch.stack(self.human_to_ver_per_human) # (n, 3)

        # human_to_ver per envs
        self.env_to_human_inds = torch.arange(self.num_envs, device=self.device) % self.n_human_types 
        self.human_to_ver_per_envs = self.human_to_ver_per_human[self.env_to_human_inds] # (N, 3)
        self.human_to_ver_per_envs_real = self.human_to_ver_per_envs * self.res # (N, 3)

        # trajectory points
        self.traj_points_np_list = [pv.read(traj_file).points for traj_file in traj_file_list]
        self.traj_points_list = [torch.tensor(self.traj_points_np, device=device)
                                           for self.traj_points_np in self.traj_points_np_list] # (n, P_t, 3)
        
        # get directions and positions
        self.get_traj_center_drct_radius()
        self.human_to_traj_pos = torch.stack(self.traj_center_list)[self.env_to_human_inds] * self.res # (N, 3)
        self.traj_drct = torch.stack(self.traj_drct_list)[self.env_to_human_inds] # (N, 3)
        self.traj_radius = torch.stack(self.traj_radius_list)[self.env_to_human_inds] * self.res # (N,)
        self.traj_half_length = torch.stack(self.traj_half_length_list)[self.env_to_human_inds] * self.res # (N,)

        # visualize
        if self.if_vis:
            self.tip_points_a = torch.zeros((self.num_envs, 3), device=self.device)
            self.tip_points_b = torch.zeros((self.num_envs, 3), device=self.device)
            self.tip_points_b[:, 2] = -20
            self.tip_points_ab = torch.stack([self.tip_points_a, self.tip_points_b], dim=1)
            self.tip_line_list = []
            for i in range(self.num_envs):
                tip_line = pv.Line(pointa=self.tip_points_a[i, :].cpu().numpy(), 
                                   pointb=self.tip_points_b[i, :].cpu().numpy(), resolution=1)
                self.tip_line_list.append(tip_line)
            self.visualize()


    def get_traj_center_drct_radius(self):
        '''
        get the direction and center of the trajectories
        '''
        self.traj_center_list = [torch.mean(traj_points, dim=0) for traj_points in self.traj_points_list]
        self.traj_drct_list = []
        self.traj_radius_list = []
        self.traj_half_length_list = []
        i = 0
        for traj_points in self.traj_points_list:
            mean = self.traj_center_list[i]
            # Center the points
            centered_points = traj_points - mean

            # Compute the covariance matrix
            covariance_matrix = (centered_points.T @ centered_points) / (centered_points.shape[0] - 1)

            # Eigen decomposition
            eigvals, eigvecs = torch.linalg.eigh(covariance_matrix)

            # Get the eigenvector with the largest eigenvalue (principal component)
            principal_axis = eigvecs[:, -1]  # Last column corresponds to largest eigenvalue
            radius_axis = eigvecs[:, 0]  # First column corresponds to smallest eigenvalue

            principal_axis = principal_axis / torch.norm(principal_axis)  # Normalize to unit vector

            # only take direction downwards
            dot_y = torch.dot(principal_axis, torch.tensor([0.0, 1.0, 0.0], device=self.device))
            if dot_y < 0:
                principal_axis = -principal_axis
            self.traj_drct_list.append(principal_axis)

            # get half length
            half_length = torch.norm(centered_points @ principal_axis.reshape(-1, 1), dim=-1).mean()
            self.traj_half_length_list.append(half_length)

            # get radius
            radius = torch.norm(centered_points, dim=-1).mean() ** 2 - half_length ** 2
            self.traj_radius_list.append(torch.sqrt(radius))

            i += 1


    def compute_tip_in_traj(self, human_to_tip_pos, human_to_tip_rot):
        '''
        compute the tip position in the trajectory
        '''
        # position
        traj_to_tip_pos = human_to_tip_pos - self.human_to_traj_pos
        pos_along_drct = torch.sum(traj_to_tip_pos * self.traj_drct, dim=-1)
        distance_to_traj = torch.norm(traj_to_tip_pos - pos_along_drct.unsqueeze(-1) * self.traj_drct, dim=-1)
        # direction
        human_to_tip_rot_mat = matrix_from_quat(human_to_tip_rot)
        tip_drct = human_to_tip_rot_mat[:, :, 2]
        cos_angle = torch.sum(self.traj_drct * tip_drct, dim=-1)

        sin_angle = torch.sqrt(torch.maximum(1 - cos_angle**2, torch.zeros_like(cos_angle)))

        return pos_along_drct, distance_to_traj, sin_angle


    def visualize(self, index=0):
        self.p = pv.Plotter()
        
        tip_points_a_np = self.tip_points_a.cpu().numpy()
        tip_points_b_np = self.tip_points_b.cpu().numpy()
        for i in range(self.num_envs):
            self.tip_line_list[i].points = np.stack([tip_points_a_np[i], 
                                                     tip_points_b_np[i]], axis=0)
            if i%self.n_human_types == index:
                self.p.add_mesh(self.tip_line_list[i], color='green', opacity=1.0)
        # add tip representation
        cylinder_points = self.traj_points_np_list[index]
        print(cylinder_points)
        print(self.human_to_traj_pos[index])
        goal_points = cylinder_points[cylinder_points[:, 1] > self.human_to_traj_pos[index, 1].item() / self.res]
        self.p.add_mesh(self.vertebra_points_np_list[index], color='cornflowerblue', point_size=0.5)
        self.p.add_mesh(goal_points, color='darkred', point_size=2.0)
        # self.p.add_mesh(pv.Sphere(center=self.human_to_traj_pos[index].cpu().numpy() / self.res, 
        #                           radius=self.traj_radius[index].item()/self.res), color='green', opacity=0.5)

        self.p.show(interactive_update=True)
        self.p.show_axes()

        
    def update_tip_vis(self, human_to_tip_pos, human_to_tip_rot):
        scaled_pos = human_to_tip_pos / self.res
        tip_points_ab = transform_points(self.tip_points_ab, scaled_pos, human_to_tip_rot)
        tip_points_a_np = tip_points_ab[:, 0, :].cpu().numpy()
        tip_points_b_np = tip_points_ab[:, 1, :].cpu().numpy()
        for i in range(self.num_envs):
            # TODO: change original points
            original_points = self.tip_line_list[i].points
            self.tip_line_list[i].points += np.stack([tip_points_a_np[i, :], 
                                                     tip_points_b_np[i, :]], axis=0) - original_points
            # print(self.tip_line_list[i].points)
            # TODO: add new points
            # if i % 5 == 0:
            #     tip_line = pv.Line(pointa=tip_points_a_np[i], 
            #                        pointb=tip_points_b_np[i], 
            #                        resolution=1)
            #     self.p.add_mesh(tip_line, color='green', opacity=0.2)
            
        self.p.update()
        

