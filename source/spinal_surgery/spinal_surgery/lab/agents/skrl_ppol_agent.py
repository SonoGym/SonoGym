from __future__ import annotations

from typing import Any

import copy
import dataclasses
import itertools

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

from spinal_surgery.lab.agents.lagrangian_optimizer import LagrangianOptimizer

import wandb


@dataclasses.dataclass(kw_only=True)
class PPOL_CFG(PPO_CFG):
    """Configuration for the PPO Lagrangian (PPOL) agent."""

    cost_value_preprocessor: type | None = None
    """Preprocessor class to process the cost value network's output."""

    cost_value_preprocessor_kwargs: dict = dataclasses.field(default_factory=dict)
    """Keyword arguments for the cost value preprocessor's constructor."""

    cost_limit: float = 2.0
    """Cost limit (target threshold) for the Lagrangian multiplier."""

    use_lagrangian: bool = True
    """Whether to use the Lagrangian formulation."""

    lagrangian_pid: tuple = (0.05, 0.0005, 0.1)
    """PID parameters for Lagrangian optimization."""

    rescaling: bool = True
    """Whether to rescale the policy loss by ``1 / (lagrangian + 1)``."""

    state_dependence: bool = True
    """Whether to use a per-state Lagrangian multiplier."""

    large_scale: float = 1
    """Scale factor applied to the per-state Lagrangian multiplier."""


# Backwards-compatible default-config dict (kept so existing training scripts that do
# ``update_config(PPOL_DEFAULT_CONFIG, agent_cfg)`` keep working).
PPOL_DEFAULT_CONFIG: dict = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},
    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},
    "cost_value_preprocessor": None,
    "cost_value_preprocessor_kwargs": {},
    "random_timesteps": 0,
    "learning_starts": 0,
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "entropy_loss_scale": 0.0,
    "value_loss_scale": 1.0,
    "kl_threshold": 0,
    "rewards_shaper": None,
    "time_limit_bootstrap": False,
    "cost_limit": 2.0,
    "use_lagrangian": True,
    "lagrangian_pid": (0.05, 0.0005, 0.1),
    "rescaling": True,
    "state_dependence": True,
    "large_scale": 1,
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
    if "lambda" in cfg and "gae_lambda" not in cfg:
        cfg["gae_lambda"] = cfg.pop("lambda")
    else:
        cfg.pop("lambda", None)
    cfg.pop("clip_predicted_values", None)
    cfg.pop("rewards_shaper_scale", None)
    cfg.pop("seed", None)
    valid_keys = {f.name for f in dataclasses.fields(PPOL_CFG)}
    return {k: v for k, v in cfg.items() if k in valid_keys}


class PPOLagrangian(PPO):
    """PPO with a Lagrangian-relaxed cost-value head for safe RL."""

    cfg: PPOL_CFG  # narrow the type from PPO's ``PPO_CFG``

    def __init__(
        self,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: PPOL_CFG | dict | None = None,
    ) -> None:
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            ppol_cfg = PPOL_CFG(**_normalize_cfg(cfg))
        else:
            ppol_cfg = cfg

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=ppol_cfg,
        )

        # cost value head
        self.cost_value = self.models.get("cost_value", None)
        self.checkpoint_modules["cost_value"] = self.cost_value

        # PPOL specific attributes
        self._use_lagrangian = self.cfg.use_lagrangian
        self._rescaling = self.cfg.rescaling
        self._cost_limit = self.cfg.cost_limit
        self._state_dependence = self.cfg.state_dependence
        self._large_scale = self.cfg.large_scale
        # convert PID to a list (the original code mutates ``[1]``/``[2]`` for the
        # global-Lagrangian path, which would fail on a tuple)
        self._lagrangian_pid = list(self.cfg.lagrangian_pid)

        # Lagrangian optimizer(s)
        if self._use_lagrangian:
            if not self._state_dependence:
                self._lagrangian_pid[1] = 0
                self._lagrangian_pid[2] = 0
                self.lag_optim = LagrangianOptimizer(tuple(self._lagrangian_pid))
            else:
                self.lag_optim = [
                    LagrangianOptimizer(tuple(self._lagrangian_pid)) for _ in range(self.cfg.mini_batches)
                ]
        else:
            self.lag_optim = []

        # Rebuild the optimizer to also include the cost value head's parameters.
        if self.cost_value is not None and self.policy is not None and self.value is not None:
            param_groups = [self.policy.parameters(), self.cost_value.parameters()]
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

        # cost value preprocessor
        if self.cfg.cost_value_preprocessor:
            self._cost_value_preprocessor = self.cfg.cost_value_preprocessor(
                **self.cfg.cost_value_preprocessor_kwargs
            )
            self.checkpoint_modules["cost_value_preprocessor"] = self._cost_value_preprocessor
        else:
            self._cost_value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: Any = None) -> None:
        super().init(trainer_cfg=trainer_cfg)

        if self.memory is not None:
            self.memory.create_tensor(name="costs", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_advantages", size=1, dtype=torch.float32)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "cost_values",
                "cost_returns",
                "cost_advantages",
            ]

    def _build_inputs(self, observations: torch.Tensor, states: torch.Tensor | None) -> dict:
        """Build the model-input dict.

        The user's models in ``skrl_actor_critic.py`` access ``inputs["states"]``. To keep
        them compatible with the new skrl convention, we expose the preprocessed
        observations under both ``"observations"`` and ``"states"`` keys.
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
        # Skip PPO's default transition recording (no awareness of costs/cost-values)
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

        # compute cost values for storage
        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = self._build_inputs(observations, states)
            cost_values, _ = self.cost_value.act(inputs, role="cost_value")
            cost_values = self._cost_value_preprocessor(cost_values, inverse=True)

        # time-limit (truncation) bootstrapping
        if self.cfg.time_limit_bootstrap:
            rewards = rewards + self.cfg.discount_factor * self._current_values * truncated
            costs = costs + self.cfg.discount_factor * cost_values * truncated

        self.memory.add_samples(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            log_prob=self._current_log_prob,
            values=self._current_values,
            cost_values=cost_values,
            costs=costs,
        )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def safety_loss(self, cost_value) -> tuple[torch.Tensor, dict]:
        """Compute the global-Lagrangian safety loss and stats."""
        lag = self.lag_optim.get_lag()
        rescaling = 1.0 / (lag + 1) if self._rescaling else 1
        stats = {"loss/rescaling": rescaling}
        loss = torch.mean(cost_value * lag)
        stats["loss/lagrangian"] = lag
        stats["loss/actor_safety"] = loss.item()
        return loss, stats

    def safety_loss_state_dep(self, cost_value) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute the per-state Lagrangian safety loss and stats."""
        lag = self.cur_lag
        rescaling = 1.0 / (lag + 1)
        stats = {"loss/rescaling": torch.mean(rescaling).item()}
        loss = torch.mean(cost_value * lag)
        stats["loss/lagrangian"] = torch.mean(lag).item()
        stats["loss/actor_safety"] = loss.item()
        return loss, stats, rescaling

    def update_lagrangian(self, cost_value) -> None:
        """Update the global Lagrangian multiplier before the policy update."""
        self.lag_optim.step(cost_value, self._cost_limit)

    def update_lagrangian_per_state(self, cost_value) -> None:
        """Compute the per-state Lagrangian multiplier before the policy update."""
        proj_cost_value = torch.clamp(cost_value - self._cost_limit, min=0)
        with torch.no_grad():
            self.cur_lag = self._lagrangian_pid[0] * proj_cost_value * self._large_scale

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
                next_values = values[i + 1] if i < memory_size - 1 else next_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # compute returns and advantages (and their cost-side counterparts)
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = self._build_inputs(self._current_next_observations.float(), self._current_next_states)
            self.value.enable_training_mode(False)
            self.cost_value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            last_cost_values, _ = self.cost_value.act(inputs, role="cost_value")
            self.value.enable_training_mode(True)
            self.cost_value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)
            last_cost_values = self._cost_value_preprocessor(last_cost_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )
        cost_values = self.memory.get_tensor_by_name("cost_values")
        cost_returns, cost_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("costs"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=cost_values,
            next_values=last_cost_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
        self.memory.set_tensor_by_name("cost_values", self._cost_value_preprocessor(cost_values, train=True))
        self.memory.set_tensor_by_name(
            "cost_returns", self._cost_value_preprocessor(cost_returns, train=True)
        )
        self.memory.set_tensor_by_name("cost_advantages", cost_advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cost_value_loss = torch.tensor(0.0, device=self.device)
        stats_actor: dict = {"loss/lagrangian": 0.0, "loss/actor_safety": 0.0, "loss/rescaling": 1.0}

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
                sampled_cost_values,
                sampled_cost_returns,
                sampled_cost_advantages,
            ) in sampled_batches:

                # update Lagrangian multiplier(s) before the policy update
                if self._state_dependence:
                    self.update_lagrangian_per_state(sampled_cost_values)
                else:
                    self.update_lagrangian(cost_values)

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
                            predicted_values - sampled_values,
                            min=-self.cfg.value_clip,
                            max=self.cfg.value_clip,
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # safety-augmented policy loss
                    if self._state_dependence:
                        loss_actor_safety, stats_actor, rescaling = self.safety_loss_state_dep(
                            sampled_cost_advantages
                        )
                    else:
                        loss_actor_safety, stats_actor = self.safety_loss(ratio * sampled_cost_advantages)
                    rescaling = stats_actor["loss/rescaling"]
                    policy_loss = rescaling * (policy_loss + loss_actor_safety)

                    # cost-value loss
                    predicted_cost_values, _ = self.cost_value.act(inputs, role="cost_value")
                    cost_value_loss = self.cfg.value_loss_scale * F.mse_loss(
                        sampled_cost_returns, predicted_cost_values
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + cost_value_loss).backward()

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
                    if self.cost_value is not None:
                        params_to_clip += list(self.cost_value.parameters())
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

        denom = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)

        self.track_data("Loss / Cost value loss", cost_value_loss.item())
        self.track_data("Loss / lag", stats_actor["loss/lagrangian"])
        self.track_data("Loss / policy safety loss", stats_actor["loss/actor_safety"])

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def track_data(self, tag: str, value: float) -> None:
        super().track_data(tag, value)
        if wandb.run is not None:
            wandb.log({tag: value})
