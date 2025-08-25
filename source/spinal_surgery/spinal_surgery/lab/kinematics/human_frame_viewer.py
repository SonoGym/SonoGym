
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, transform_points
import pyvista as pv
import torch
import numpy as np




class HumanFrameViewer:
    # Class: human frame viewer:
    # 1 plotter per label map
    # Visualize all end effectors for each label map (2 axis for us frame)
    # Function: slice label map
    # Function: update plotter

    def __init__(self, label_maps, num_envs, device, label_res=0.0015, height_img=0.1, visualize=True, plane_axes={'h': [0, 0, 1], 'w': [1, 0, 0]}):
        '''
        label maps: list of label maps (3D volumes)
        num_envs: number of environments
        plane_axes: dict of plane axes for imaging, in our case is 'x' and 'z' axes of the ee frame
        '''
        
        self.label_maps = [torch.tensor(label_map, dtype=torch.uint8, device=device) for label_map in label_maps]
        self.num_envs = num_envs
        self.n_human_types = len(label_maps)
        self.plane_axes = plane_axes
        self.label_res = label_res
        self.height_img = height_img

        if visualize: 
            self.p_list = []
            # add label maps
            for i in range(self.n_human_types):
                p = pv.Plotter()
                self.p_list.append(p)

                p.add_volume(label_maps[i], opacity=0.01)
                p.add_mesh(pv.Sphere(radius=5.0), color='red')
                p.show_axes()
                p.show(interactive_update=True)

            # add origin markers (num_envs)
            self.w_markers_list = []
            self.h_markers_list = []

            self.w_axis = np.array(self.plane_axes['w'])
            self.h_axis = np.array(self.plane_axes['h'])

            self.w_axis_tensor = torch.tensor(self.plane_axes['w'], device=device)
            self.h_axis_tensor = torch.tensor(self.plane_axes['h'], device=device)

            self.np_linear_space = np.linspace(0.0, 10, 5).reshape((-1, 1))
            
            for j in range(num_envs):
                # create axis
                
                ee_w_points = self.np_linear_space * self.w_axis.reshape((1, -1)) + height_img / self.label_res * self.h_axis.reshape((1, -1))
                ee_h_points = (self.np_linear_space + height_img / self.label_res) * self.h_axis.reshape((1, -1))
                w_axis_pv = pv.PolyData(ee_w_points)
                h_axis_pv = pv.PolyData(ee_h_points)
                
                # save to the corresponding list
                self.w_markers_list.append(w_axis_pv)
                self.h_markers_list.append(h_axis_pv)

                self.p_list[j % self.n_human_types].add_mesh(w_axis_pv, color='red')
                self.p_list[j % self.n_human_types].add_mesh(h_axis_pv, color='green')


        

    def ee_pose_to_human_pcd(self, human_to_ee_pos, human_to_ee_quat):
        '''
        use point cloud to represent poses
        - human_to_ee_poses: tensor of human to ee poses: num_envs x 3
        - human_to_ee_quats: tensor of human to ee quaternions: num_envs x 4
        '''
        # TODO: allocate once in the init
        
        ee_w_points = (torch.linspace(0.0, 10, 5, device=human_to_ee_pos.device).reshape((-1, 1)) * self.w_axis_tensor.reshape((1, -1))
                       + self.height_img / self.label_res * self.h_axis_tensor.reshape((1, -1)))
        ee_h_points = (torch.linspace(0.0, 10, 5, device=human_to_ee_pos.device).reshape((-1, 1)) 
                       + self.height_img / self.label_res) * self.h_axis_tensor.reshape((1, -1))
        human_w_points = transform_points(ee_w_points, human_to_ee_pos / self.label_res, human_to_ee_quat)
        human_h_points = transform_points(ee_h_points, human_to_ee_pos / self.label_res, human_to_ee_quat)

        return human_w_points, human_h_points # num_envs x 40 x 3

    
    def update_plotter(self, 
                       world_human_pos, 
                       world_human_quat, 
                       world_ee_pos,
                       world_ee_quat):
        '''
        world_human_pos: tensor of human positions in world frame: num_envs x 3
        world_human_quat: tensor of human quaternions in world frame: num_envs x 4
        world_ee_pos: tensor of end effector positions in world frame: num_envs x 3
        world_ee_quat: tensor of end effector quaternions in world frame: num_envs x 4
        '''
        human_ee_pos, human_ee_quat = subtract_frame_transforms(
            world_human_pos, world_human_quat, world_ee_pos, world_ee_quat
        )

        # obtain point positions in human frame
        human_w_points, human_h_points = self.ee_pose_to_human_pcd(human_ee_pos, human_ee_quat)
        human_w_points = human_w_points.cpu().numpy()
        human_h_points = human_h_points.cpu().numpy()

        # update markers
        for i in range(self.num_envs):
            self.w_markers_list[i].points = human_w_points[i, :, :]
            self.h_markers_list[i].points = human_h_points[i, :, :]
        
        for p in self.p_list:
            p.update()