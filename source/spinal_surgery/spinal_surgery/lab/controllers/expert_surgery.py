import torch
from isaaclab.utils.math import matrix_from_quat, quat_inv


class ExpertSurgery:
    def __init__(self, max_action, action_scale, ratio, device):
        self.max_action = max_action
        self.action_scale = action_scale
        self.device = device
        self.ratio = torch.tensor(ratio, device=device).reshape(1, -1)

    def batched_angle_axis_from_vectors(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Computes angle-axis rotation vectors that rotate a[i] to b[i] for each i in a batch.
        a, b: (N, 3) tensors
        Returns: (N, 3) tensor of angle-axis vectors
        """
        a_norm = a / a.norm(dim=1, keepdim=True)
        b_norm = b / b.norm(dim=1, keepdim=True)

        cross_prod = torch.cross(a_norm, b_norm, dim=1)
        dot_prod = (a_norm * b_norm).sum(dim=1)

        cross_norm = cross_prod.norm(dim=1, keepdim=True)
        eps = 1e-6

        # Initialize angle-axis output
        angle_axis = torch.zeros_like(a)

        # Case 1: Regular case
        valid = cross_norm.squeeze(-1) >= eps
        axis = torch.zeros_like(a)
        axis[valid] = cross_prod[valid] / cross_norm[valid]
        angle = torch.acos(torch.clamp(dot_prod, -1.0, 1.0)).unsqueeze(-1)
        angle_axis[valid] = angle[valid] * axis[valid]

        # Case 2: a and b are opposite -> rotate 180 degrees around orthogonal axis
        opposite = (cross_norm.squeeze(-1) < eps) & (dot_prod < 0)
        if opposite.any():
            a_opp = a_norm[opposite]
            # Choose reference vectors that are not colinear with a
            ref = torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a_opp)
            same = torch.isclose(a_opp, ref, atol=1e-3).all(dim=1)
            ref[same] = torch.tensor([0.0, 1.0, 0.0], device=a.device)

            axis_opp = torch.cross(a_opp, ref, dim=1)
            axis_opp = axis_opp / axis_opp.norm(dim=1, keepdim=True)
            angle_axis[opposite] = torch.pi * axis_opp

        return angle_axis
        
    def get_action(self, info):
        human_traj_drct = info['traj_drct']
        human_to_tip_pos = info['human_to_tip_pos']
        human_to_tip_quat = info['human_to_tip_quat']
        safe_height = info['safe_height'].reshape(-1, 1)
        traj_half_length = info['traj_half_length'].reshape(-1, 1)
        traj_radius = info['traj_radius']
        tip_to_traj_dist = info['tip_to_traj_dist']
        traj_to_tip_sin = info['traj_to_tip_sin']
        human_to_traj_pos = info['human_to_traj_pos']
        tip_pos_along_traj = info['tip_pos_along_traj']

        # first goal position in human frame
        goal_pos_1 = human_to_traj_pos - human_traj_drct * safe_height
        goal_pos_2 = human_to_traj_pos + human_traj_drct * traj_half_length
        
        human_pos_diff_1 = goal_pos_1 - human_to_tip_pos
        human_pos_diff_2 = goal_pos_2 - human_to_tip_pos

        human_to_tip_rot_mat = matrix_from_quat(human_to_tip_quat)
        human_to_tip_axis = human_to_tip_rot_mat[:, :, 2]
        # get angle axis distance
        human_rot_diff = self.batched_angle_axis_from_vectors(
            human_to_tip_axis, human_traj_drct
        )
        # convert to tip frame
        tip_to_human_quat = quat_inv(human_to_tip_quat)
        tip_to_human_rot_mat = matrix_from_quat(tip_to_human_quat)

        tip_pos_diff_1 = torch.bmm(
            tip_to_human_rot_mat, human_pos_diff_1.unsqueeze(-1)
        ).squeeze(-1)
        tip_pos_diff_2 = torch.bmm(
            tip_to_human_rot_mat, human_pos_diff_2.unsqueeze(-1)
        ).squeeze(-1)

        tip_rot_diff = torch.bmm(
            tip_to_human_rot_mat, human_rot_diff.unsqueeze(-1)
        ).squeeze(-1)

        # check if along the trajectory
        along_traj = torch.logical_and(
            tip_to_traj_dist < traj_radius,
            traj_to_tip_sin < 0.2
        )
        safety_critical = tip_pos_along_traj.reshape((-1,)) > - safe_height.reshape((-1,))
        # TODO: for collect dataset
        insert_goal = torch.logical_or(along_traj, safety_critical)
        # TODO for real expert:
        # insert_goal = along_traj
        outside_patient = torch.logical_not(insert_goal)

        tip_pos_diff = torch.zeros_like(tip_pos_diff_1, device=self.device)
        tip_pos_diff[outside_patient] = tip_pos_diff_1[outside_patient]
        tip_pos_diff[insert_goal] = tip_pos_diff_2[insert_goal]

        # TODO: debug
        # tip_pos_diff = torch.zeros_like(tip_pos_diff, device=self.device)
        action = torch.cat([
            tip_pos_diff,
            tip_rot_diff,
        ], dim=-1)
        # scale the action
        action = torch.clamp(action * self.ratio, -self.action_scale, self.action_scale)
        
        return action