import torch
import pyvista as pv

class VertebraViewer:
    def __init__(self, num_envs, n_human_types, vertebra_file_list, device):
        self.num_envs = num_envs
        self.n_human_types = n_human_types
        self.vertebra_file_list = vertebra_file_list
        self.device = device

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

