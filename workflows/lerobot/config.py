import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common import envs, policies  # noqa: F401
from lerobot.common.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None

    seed: int | None = 1000

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
