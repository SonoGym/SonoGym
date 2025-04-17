# expert controller that 
import torch
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat


class ExpertGuidance:
    def __init__(self, max_action, action_scale, ratio, device):
        self.max_action = max_action
        self.action_scale = action_scale
        self.device = device
        self.ratio = torch.tensor(ratio, device=device).reshape(1, -1)

    def get_action(self, info):
        human_to_ee_pos = info["human_to_ee_pos"]  # (N, 3)
        human_to_ee_quat = info["human_to_ee_quat"]  # (N, 4)
        cur_cmd_pose = info["cur_cmd_pose"]  # (N, 3)
        goal_cmd_pose = info["goal_cmd_pose"]  # (N, 3)

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
            cmd_diff[:, 2]
        ], dim=-1)

        # scale the action
        action = torch.clamp(ee_cmd_diff * self.ratio, -self.action_scale, self.action_scale)

        return action
        




