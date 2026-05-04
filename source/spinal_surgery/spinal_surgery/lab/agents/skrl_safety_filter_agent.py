from __future__ import annotations

from typing import Any

import copy
import dataclasses
import itertools

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from skrl import config, logger
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

import wandb


@dataclasses.dataclass(kw_only=True)
class SPPO_CFG(PPO_CFG):
    """Configuration for the Safety-Filter PPO agent (SPPO)."""

    cost_limit: float = 2.0
    """Cost limit for Lagrangian/safety filtering."""

    polyak: float = 0.01
    """Polyak averaging coefficient (reserved for target networks)."""

    learn_cost: bool = True
    """Whether to learn the cost critic. If True, ``cost_limit`` is overridden to 0.5."""


# Backwards-compatible default-config dict (kept so existing training scripts that
# do ``update_config(SPPO_DEFAULT_CONFIG, agent_cfg)`` keep working).
SPPO_DEFAULT_CONFIG: dict = {
    "rollouts": 16,
    "learning_epochs": 32,
    "mini_batches": 5,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 1e-4,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 1.0,
    "ratio_clip": 0.2,
    "value_clip": 0.1,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 1.0,
    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,
    "cost_limit": 2.0,
    "polyak": 0.01,
    "learn_cost": True,
    "mixed_precision": False,
    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}


def _normalize_cfg(cfg: dict) -> dict:
    """Map skrl-1.x config keys to skrl-2.x equivalents and drop removed keys."""
    cfg = copy.deepcopy(cfg)
    # 1.x -> 2.x renames
    if "lambda" in cfg and "gae_lambda" not in cfg:
        cfg["gae_lambda"] = cfg.pop("lambda")
    else:
        cfg.pop("lambda", None)
    # removed in 2.x (clipping is now controlled by ``value_clip > 0``)
    cfg.pop("clip_predicted_values", None)
    # not part of PPO_CFG (kept for user bookkeeping only)
    cfg.pop("rewards_shaper_scale", None)
    cfg.pop("seed", None)
    # keep only fields that SPPO_CFG knows about; preserve ``experiment`` (handled by Agent base)
    valid_keys = {f.name for f in dataclasses.fields(SPPO_CFG)}
    return {k: v for k, v in cfg.items() if k in valid_keys}


class SafetyFilterPPO(PPO):
    """PPO variant with an additional Q-style cost critic for safety filtering."""

    cfg: SPPO_CFG  # narrow the type from PPO's ``PPO_CFG``

    def __init__(
        self,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: SPPO_CFG | dict | None = None,
    ) -> None:
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            sppo_cfg = SPPO_CFG(**_normalize_cfg(cfg))
        else:
            sppo_cfg = cfg

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=sppo_cfg,
        )

        # cost critic
        self.cost_critic = self.models.get("cost_critic", None)
        self.checkpoint_modules["cost_critic"] = self.cost_critic

        # safety-filter specific attributes
        self._cost_limit = self.cfg.cost_limit
        self._polyak = self.cfg.polyak
        self._learn_cost = self.cfg.learn_cost
        if self._learn_cost:
            self._cost_limit = 0.5

        # rebuild the optimizer to also include the cost critic's parameters.
        # ``self.cfg.learning_rate`` and ``self.cfg.learning_rate_scheduler`` are expanded
        # to tuples by ``PPO_CFG.expand()`` during the base ``__init__``.
        if self.cost_critic is not None and self.policy is not None and self.value is not None:
            param_groups = [self.policy.parameters(), self.cost_critic.parameters()]
            if self.policy is not self.value:
                param_groups.insert(1, self.value.parameters())
            lr = self.cfg.learning_rate[0]  # type: ignore[index]
            self.optimizer = torch.optim.Adam(itertools.chain(*param_groups), lr=lr)
            self.checkpoint_modules["optimizer"] = self.optimizer

            scheduler_cls = self.cfg.learning_rate_scheduler[0]  # type: ignore[index]
            if scheduler_cls is not None:
                scheduler_kwargs = self.cfg.learning_rate_scheduler_kwargs[0]  # type: ignore[index]
                self.scheduler = scheduler_cls(self.optimizer, **scheduler_kwargs)
            else:
                self.scheduler = None

    def init(self, *, trainer_cfg: Any = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        if self.memory is not None:
            self.memory.create_tensor(name="costs", size=1, dtype=torch.float32)
            # Tensors sampled during training. Mirrors PPO's order with ``costs`` appended.
            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "costs",
            ]

    def _build_inputs(self, observations: torch.Tensor, states: torch.Tensor | None) -> dict:
        """Build the model-input dict.

        The user's models in ``skrl_actor_critic.py`` access ``inputs["states"]``. To keep
        them compatible with the new skrl convention (where observations are the policy
        input), we expose the preprocessed observations under both ``"observations"`` and
        ``"states"`` keys. If a real ``states`` tensor is provided (e.g. centralized
        critic), it is preprocessed normally.
        """
        preprocessed_obs = self._observation_preprocessor(observations)
        if states is None:
            preprocessed_states = preprocessed_obs
        else:
            preprocessed_states = self._state_preprocessor(states)
        return {"observations": preprocessed_obs, "states": preprocessed_states}

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        inputs = self._build_inputs(observations, states)

        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            actions, outputs = self.policy.act(inputs, role="policy")
            self._current_log_prob = outputs["log_prob"]

            if self.training:
                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)

        # Safety filter (kept disabled for now — preserves the original commented-out behavior).
        # taken_actions = actions.clone()
        # with torch.no_grad():
        #     q_values, _ = self.cost_critic.act({**inputs, "taken_actions": actions}, role="cost_critic")
        # taken_actions[q_values.squeeze(-1) > self._cost_limit] = 0.0

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor | None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        # Skip PPO's default transition recording (it doesn't know about ``costs``)
        # and call the base ``Agent.record_transition`` for tracking-only side effects.
        super(PPO, self).record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if not self.training:
            return

        self._current_next_observations = next_observations
        self._current_next_states = next_states

        costs = infos["cost"].reshape((-1, 1))

        if self.cfg.rewards_shaper is not None:
            rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

        if self.cfg.time_limit_bootstrap:
            rewards = rewards + self.cfg.discount_factor * self._current_values * truncated

        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            log_prob=self._current_log_prob,
            values=self._current_values,
            costs=costs,
        )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = self._build_inputs(self._current_next_observations.float(), self._current_next_states)
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cost_q_loss = torch.tensor(0.0, device=self.device)

        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_costs,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    preprocessed_obs = self._observation_preprocessor(sampled_observations, train=not epoch)
                    if sampled_states is None or sampled_states.numel() == 0:
                        preprocessed_states = preprocessed_obs
                    else:
                        preprocessed_states = self._state_preprocessor(sampled_states, train=not epoch)
                    inputs = {"observations": preprocessed_obs, "states": preprocessed_states}

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # cost Q loss
                    target_q = sampled_costs
                    q_cost, _ = self.cost_critic.act(
                        {**inputs, "taken_actions": sampled_actions}, role="cost_critic"
                    )
                    cost_q_loss = self.cfg.value_loss_scale * F.mse_loss(target_q, q_cost)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + cost_q_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        params_to_clip = list(self.policy.parameters())
                    else:
                        params_to_clip = list(itertools.chain(self.policy.parameters(), self.value.parameters()))
                    if self.cost_critic is not None:
                        params_to_clip += list(self.cost_critic.parameters())
                    nn.utils.clip_grad_norm_(params_to_clip, self.cfg.grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # tracking
        denom = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)

        self.track_data("Loss / Cost Q value loss", cost_q_loss.item())
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def track_data(self, tag: str, value: float) -> None:
        super().track_data(tag, value)
        if wandb.run is not None:
            wandb.log({tag: value})
