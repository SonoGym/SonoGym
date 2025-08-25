import torch
from spinal_surgery.lab.kinematics.utils import *
import os
from isaaclab.utils.math import combine_frame_transforms, transform_points, subtract_frame_transforms, matrix_from_quat
from spinal_surgery.lab.kinematics.surface_motion_planner import quat_from_matrix_optimize

class GTMotionGenerator:

    def __init__(self, goal_cmd_pose, scale, num_envs, surface_map_list, surface_normal_list, label_res, US_height):
        '''
        goal_cmd_pose: (n_human, 3): x, z, angle
        surface_map_list: list of surface map list (n_human, X*Z), store the height of the skin
        surface_normal_list: list of surface normal list (n_human, X*Z, 3), store the normal of the skin
        device: device
        '''
        
        self.n_human_types = len(surface_map_list)
        self.surface_map_list = surface_map_list
        self.surface_normal_list = surface_normal_list
        self.num_envs = num_envs
        self.goal_cmd_pose = goal_cmd_pose
        self.scale = scale
        self.label_res = label_res
        self.height = US_height

        
        pass

    def generate_gt_human_cmd(self, cur_cmd_pose):
        '''
        generate grount truth human command from current human command pose
        cur_cmd_pose: (num_envs, 3): x, z, angle
        cur_human_ee_pos: (num_envs, 3)
        cur_human_ee_quat: (num_envs, 4)

        return:
        new_cmd_pose: (num_envs, 3): x, z, angle
        new_human_ee_pos: (num_envs, 3)
        new_human_ee_quat: (num_envs, 4)
        '''
        cmd_diff = self.goal_cmd_pose - cur_cmd_pose
        cmd_diff[:, 0:2] = cmd_diff[:, 0:2] / (torch.norm(cmd_diff[:, 0:2], dim=-1, keepdim=True) + 1e-6)
        cmd_diff[:, 0:2] *= self.scale
        cmd_diff[:, 2] = cmd_diff[:, 2] / (torch.abs(cmd_diff[:, 2]) + 1e-6) * self.scale / 50


        return cmd_diff, cmd_diff + cur_cmd_pose
    

    def compute_gt_ee_cmd(self, gt_world_target_ee_pos, gt_world_target_ee_quat, cur_world_ee_pos, cur_world_ee_quat):
        '''
        compute gt motion in ee frame
        human_target_ee_pos: (num_envs, 3)
        human_target_ee_quat: (num_envs, 4)
        cur_human_ee_pos: (num_envs, 3)
        cur_human_ee_quat: (num_envs, 4)

        return:
        gt_motion_ee_pos: (num_envs, 3)
        gt_motion_ee_quat: (num_envs, 4)
        '''
        ee_to_target_pos, ee_to_target_quat = subtract_frame_transforms(
            cur_world_ee_pos, cur_world_ee_quat, gt_world_target_ee_pos, gt_world_target_ee_quat
        )
        return ee_to_target_pos, ee_to_target_quat
    
        
    def human_cmd_state_from_ee_pose(self, human_to_target_pos, human_to_target_quat):
        '''
        convert ee_pose to human command'''
        
        rot_mat = matrix_from_quat(human_to_target_quat)
        z_axis = rot_mat[:, :, 2] # (num_envs, 3)
        x_axis = rot_mat[:, :, 0] # (num_envs, 3)
        angle = torch.atan2(x_axis[:, 2], x_axis[:, 0]) # (num_envs)
        pos = human_to_target_pos + z_axis * self.height
        x_z = pos[:, [0, 2]] / self.label_res

        return torch.cat([x_z, angle.unsqueeze(-1)], dim=-1) # (num_envs, 3)
    
    def generate_gt_ee_cmd_from_current_pose(self, US_slicer, cur_world_ee_pos, cur_world_ee_quat, world_human_pos, world_human_quat):
        '''
        generate gt ee command from current pose
        '''
        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            world_human_pos, world_human_quat, cur_world_ee_pos, cur_world_ee_quat
        )

        # convert current pose to human command pose
        cur_cmd_pose = self.human_cmd_state_from_ee_pose(cur_human_ee_pos, cur_human_ee_quat)
        print('cur_cmd_pose', cur_cmd_pose[:10,:])

        # compute gt human command
        gt_cmd_diff, gt_cmd_pose = self.generate_gt_human_cmd(cur_cmd_pose)

        # convert gt human command to gt ee pose
        US_slicer.update_cmd(gt_cmd_pose-US_slicer.current_x_z_x_angle_cmd)
        gt_world_target_ee_pos, gt_world_target_ee_quat = US_slicer.compute_world_ee_pose_from_cmd(
            world_human_pos, world_human_quat)

        # compute gt ee command
        gt_ee_target_pos, gt_ee_target_quat = self.compute_gt_ee_cmd(gt_world_target_ee_pos, gt_world_target_ee_quat, 
                                                       cur_world_ee_pos, cur_world_ee_quat)
        
        return gt_ee_target_pos, gt_ee_target_quat
    

    def compute_human_cmd_from_current_ee_cmd(self, cur_world_ee_pos, cur_world_ee_quat, ee_target_pos, ee_target_quat, world_human_pos, world_human_quat):
        '''
        compute human command from current ee command
        '''
        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            world_human_pos, world_human_quat, cur_world_ee_pos, cur_world_ee_quat
        )

        target_human_ee_pos, target_human_ee_quat = combine_frame_transforms(
            cur_human_ee_pos, cur_human_ee_quat, ee_target_pos, ee_target_quat
        )

        # convert ee command to human command
        target_cmd_pose = self.human_cmd_state_from_ee_pose(target_human_ee_pos, target_human_ee_quat)

        return target_cmd_pose
    



class GTDiscreteMotionGenerator:

    def __init__(self, goal_cmd_pose, scale, num_envs, surface_map_list, surface_normal_list, label_res, US_height):
        '''
        goal_cmd_pose: (n_human, 3): x, z, angle
        scale: step scale [dx, dz, dangle]
        surface_map_list: list of surface map list (n_human, X*Z), store the height of the skin
        surface_normal_list: list of surface normal list (n_human, X*Z, 3), store the normal of the skin
        label_res: label resolution
        US_height: height of the ultrasound sensor
        device: device
        '''
        
        self.n_human_types = len(surface_map_list)
        self.surface_map_list = surface_map_list
        self.surface_normal_list = surface_normal_list
        self.num_envs = num_envs
        self.goal_cmd_pose = goal_cmd_pose
        self.scale = scale
        self.label_res = label_res
        self.height = US_height

        
        pass

    def generate_gt_human_cmd(self, cur_cmd_pose):
        '''
        generate grount truth human command from current human command pose
        cur_cmd_pose: (num_envs, 3): x, z, angle

        return:
        new_cmd_pose: (num_envs, 3): x, z, angle
        new_human_ee_pos: (num_envs, 3)
        new_human_ee_quat: (num_envs, 4)
        '''
        cmd_diff = ((self.goal_cmd_pose - cur_cmd_pose) >=0 ).float() * 2 - 1
        cmd_diff *= self.scale

        return cmd_diff, cmd_diff + cur_cmd_pose
    
    def generate_gt_2d_cmd_in_ee(self, cur_cmd_pose, human_to_ee_pos, human_to_ee_quat):
        '''
        generate grount truth human command from current human command pose
        cur_cmd_pose: (num_envs, 3): x, z, angle

        return:
        new_cmd_pose: (num_envs, 3): x, z, angle
        new_human_ee_pos: (num_envs, 3)
        new_human_ee_quat: (num_envs, 4)
        '''
        cmd_diff = (self.goal_cmd_pose - cur_cmd_pose)
        # cmd_diff *= self.scale

        cmd_diff_3d = torch.stack([cmd_diff[:, 0], torch.zeros_like(cmd_diff[:, 0]), cmd_diff[:, 1]], dim=-1) # xyz
        zero_pos = torch.zeros_like(human_to_ee_pos)
        zero_quat = torch.zeros_like(human_to_ee_quat)
        zero_quat[:, 0] = 1
        ee_to_human_pos, ee_to_human_quat = subtract_frame_transforms(
            human_to_ee_pos, human_to_ee_quat, zero_pos, zero_quat
        )
        cmd_diff_ee_pos = transform_points(cmd_diff_3d.unsqueeze(1), ee_to_human_pos, ee_to_human_quat)
        cmd_diff_ee_pos = cmd_diff_ee_pos.squeeze(1)

        cmd_diff_ee = torch.cat([cmd_diff_ee_pos[:, 0:2], cmd_diff[:, 2:3]], dim=-1)
        cmd_diff_ee_sign = torch.sign(cmd_diff_ee) * self.scale
        large_index = abs(cmd_diff_ee) > abs(cmd_diff_ee_sign)
        cmd_diff_ee[large_index] = cmd_diff_ee_sign[large_index]
        
        return cmd_diff_ee
    

    def compute_gt_ee_cmd(self, gt_world_target_ee_pos, gt_world_target_ee_quat, cur_world_ee_pos, cur_world_ee_quat):
        '''
        compute gt motion in ee frame
        human_target_ee_pos: (num_envs, 3)
        human_target_ee_quat: (num_envs, 4)
        cur_human_ee_pos: (num_envs, 3)
        cur_human_ee_quat: (num_envs, 4)

        return:
        gt_motion_ee_pos: (num_envs, 3)
        gt_motion_ee_quat: (num_envs, 4)
        '''
        ee_to_target_pos, ee_to_target_quat = subtract_frame_transforms(
            cur_world_ee_pos, cur_world_ee_quat, gt_world_target_ee_pos, gt_world_target_ee_quat
        )
        return ee_to_target_pos, ee_to_target_quat
    
        
    def human_cmd_state_from_ee_pose(self, human_to_target_pos, human_to_target_quat):
        '''
        convert ee_pose to human command'''
        
        rot_mat = matrix_from_quat(human_to_target_quat)
        z_axis = rot_mat[:, :, 2] # (num_envs, 3)
        x_axis = rot_mat[:, :, 0] # (num_envs, 3)
        angle = torch.atan2(x_axis[:, 2], x_axis[:, 0]) # (num_envs)
        angle[angle < 0] += 2 * 3.1415926
        pos = human_to_target_pos + z_axis * self.height
        x_z = pos[:, [0, 2]] / self.label_res

        return torch.cat([x_z, angle.unsqueeze(-1)], dim=-1) # (num_envs, 3)
    
    def generate_gt_cmd_from_current_pose(self, US_slicer, cur_world_ee_pos, cur_world_ee_quat, world_human_pos, world_human_quat):
        '''
        generate gt ee command from current pose
        '''
        cur_human_ee_pos, cur_human_ee_quat = subtract_frame_transforms(
            world_human_pos, world_human_quat, cur_world_ee_pos, cur_world_ee_quat
        )

        # convert current pose to human command pose
        cur_cmd_pose = self.human_cmd_state_from_ee_pose(cur_human_ee_pos, cur_human_ee_quat)
        print('cur_cmd_pose', cur_cmd_pose[:10,:])

        # compute gt human command
        gt_cmd_diff, gt_cmd_pose = self.generate_gt_human_cmd(cur_cmd_pose)

        # convert gt human command to gt ee pose
        
        return gt_cmd_diff, gt_cmd_pose
    