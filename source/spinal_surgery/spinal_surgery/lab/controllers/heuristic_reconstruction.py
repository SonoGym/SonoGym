# expert controller that 
import torch
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat


class HeuristicReconstruction:
    def __init__(self, max_action, action_scale, human_pos_2d_min, human_pos_2d_max, num_sections, total_steps, ratio, device):
        '''
        human_pos_2d_min: (num_envs, 2)
        human_pos_2d_max: (num_envs, 2)
        num_sections: number of sections of the z shape trajectory
        '''
        self.human_pos_2d_min = human_pos_2d_min
        self.human_pos_2d_max = human_pos_2d_max
        self.num_sections = num_sections
        self.total_steps = total_steps

        self.max_action = max_action
        self.action_scale = action_scale
        self.device = device
        self.ratio = torch.tensor(ratio, device=device).reshape(1, -1)

        # get way points
        # get switch steps
        self.get_way_points()
        self.get_switch_steps()

    def get_way_points(self):
        '''
        define the way points of the z shape trajectory
        '''
        # get the 2d way points
        human_pos_2d_min = self.human_pos_2d_min
        human_pos_2d_max = self.human_pos_2d_max

        # if num_sections = 4, have 8 way points:
        '''
        ---- along x
        | along z

        0    3----4    7
        |    |    |    |
        |    |    |    |
        1----2    5----6
        '''
        way_points = []
        for i in range(self.num_sections + 1):
            if_start_from_min_z_side = i % 2 == 0

            way_points.append(
                torch.stack([
                    human_pos_2d_min[:, 0] + (human_pos_2d_max[:, 0] - human_pos_2d_min[:, 0]) * i / self.num_sections,
                    human_pos_2d_min[:, 1] + (human_pos_2d_max[:, 1] - human_pos_2d_min[:, 1]) * float(if_start_from_min_z_side),
                    torch.ones_like(human_pos_2d_min[:, 1], device=self.device) * 1.57  # (yaw angle)
                ], dim=-1)
            )

            # starting point
            way_points.append(
                torch.stack([
                    human_pos_2d_min[:, 0] + (human_pos_2d_max[:, 0] - human_pos_2d_min[:, 0]) * i / self.num_sections,
                    human_pos_2d_min[:, 1] + (human_pos_2d_max[:, 1] - human_pos_2d_min[:, 1]) * float(if_start_from_min_z_side),
                    torch.zeros_like(human_pos_2d_min[:, 1], device=self.device)  # (yaw angle)
                ], dim=-1)
            )

            # ending point
            way_points.append(
                torch.stack([
                    human_pos_2d_min[:, 0] + (human_pos_2d_max[:, 0] - human_pos_2d_min[:, 0]) * i / self.num_sections,
                    human_pos_2d_max[:, 1] + (human_pos_2d_min[:, 1] - human_pos_2d_max[:, 1]) * float(if_start_from_min_z_side),
                                        torch.zeros_like(human_pos_2d_min[:, 1], device=self.device)  # (yaw angle)
                ], dim=-1)
            )

            # ending point
            way_points.append(
                torch.stack([
                    human_pos_2d_min[:, 0] + (human_pos_2d_max[:, 0] - human_pos_2d_min[:, 0]) * i / self.num_sections,
                    human_pos_2d_max[:, 1] + (human_pos_2d_min[:, 1] - human_pos_2d_max[:, 1]) * float(if_start_from_min_z_side),
                    torch.ones_like(human_pos_2d_min[:, 1], device=self.device) * 1.57  # (yaw angle)
                ], dim=-1)
            )

        self.way_points = way_points
    
    def get_switch_steps(self):
        '''
        get switch time for the waypoints:
        each segment along z take num_sections unit time
        each segment along x take 1 unit time.
        '''
        num_init_segments = 1  # initial segment to go to the first way point
        num_rot_segments = (self.num_sections + 1) * 2
        num_z_segments = self.num_sections + 1
        num_x_segments = self.num_sections
        unit_steps = self.total_steps / (num_init_segments 
                                         + num_z_segments * self.num_sections 
                                         + num_x_segments 
                                         + num_rot_segments * self.num_sections)

        switch_steps = [0]
        for i in range(len(self.way_points)):
            if i == 0:
                switch_steps.append(int(unit_steps * num_init_segments))
            elif i % 4 == 0:
                switch_steps.append(int(switch_steps[-1] + unit_steps))
            else:
                switch_steps.append(int(switch_steps[-1] + unit_steps * self.num_sections))

        self.switch_steps = switch_steps

    def get_action_given_goal(self, info, goal_cmd_pose):
        human_to_ee_pos = info["human_to_ee_pos"]  # (N, 3)
        human_to_ee_quat = info["human_to_ee_quat"]  # (N, 4)
        cur_cmd_state = info["cur_cmd_state"]  # (N, 4)

        cur_cmd_pose = cur_cmd_state[:, :3]  # (N, 3)
        # compute 2d diff
        cmd_diff = goal_cmd_pose - cur_cmd_pose  # (N, 3)
        # project to the ee frame
        ee_to_human_pos, ee_to_human_quat = subtract_frame_transforms(
            human_to_ee_pos, human_to_ee_quat,
            torch.zeros_like(human_to_ee_pos, device=self.device),
            torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(human_to_ee_pos.shape[0], 1),
        )
        ee_to_human_rot_mat = matrix_from_quat(ee_to_human_quat)

        # compute pose in ee frame
        cmd_diff_pos = torch.stack(
            [cmd_diff[:, 0], torch.zeros_like(cmd_diff[:, 0], device=self.device), cmd_diff[:, 1]], dim=-1
        )
        ee_cmd_diff_pos = torch.bmm(
            ee_to_human_rot_mat, cmd_diff_pos.unsqueeze(-1)
        ).squeeze(-1)  # (N, 3)

        ee_cmd_diff = torch.stack([
            ee_cmd_diff_pos[:, 0],
            ee_cmd_diff_pos[:, 1],
            cmd_diff[:, 2],
            torch.zeros_like(cmd_diff[:, 2], device=self.device)  # (yaw angle)
        ], dim=-1)

        # scale the action
        action = ee_cmd_diff * self.ratio / self.action_scale

        return action

    def get_action(self, info, step):
        
        for i in range(len(self.switch_steps) - 1):
            if step >= self.switch_steps[i]:
                way_point_index = i

        # if way_point_index == 1:
        #     print(step)
        
        cur_goal_pose = self.way_points[way_point_index]

        action = self.get_action_given_goal(info, cur_goal_pose)

        return action
        




