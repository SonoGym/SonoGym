from __future__ import annotations

from typing import Type

from skrl.utils.runner.torch import Runner


class RNNRunner(Runner):
    """skrl Runner variant that swaps the standard PPO for ``PPO_RNN``.

    All other component lookups (memories, models, trainers, other agents) and the
    full configuration pipeline are inherited unchanged from :class:`skrl.utils.runner.torch.Runner`.
    """

    def _component(self, name: str) -> Type:
        # only override the "ppo" mapping; everything else delegates to the parent
        if name.lower() == "ppo":
            from skrl.agents.torch.ppo import PPO_RNN
            return PPO_RNN
        return super()._component(name)
