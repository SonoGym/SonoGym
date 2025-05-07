from typing import Union, Tuple, Dict, Any, Optional, List

import gymnasium
import itertools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config, logger
from packaging import version
import numpy as np

from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from spinal_surgery.lab.agents.lagrangian_optimizer import LagrangianOptimizer
import wandb


PPOL_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "cost_value_preprocessor": None,       # cost value preprocessor class (see skrl.resources.preprocessors)
    "cost_value_preprocessor_kwargs": {},  # cost value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "cost_limit": 2.0,            # cost limit for Lagrangian multiplier optimization
    "use_lagrangian": True,        # whether to use Lagrangian
    "lagrangian_pid": (0.05, 0.0005, 0.1),  # PID parameters for Lagrangian
    "rescaling": True,             # whether to rescale the Lagrangian multiplier
    "state_dependence": True,        # whether to use state-dependent Lagrangian multiplier
    "large_scale": 1,

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


class PPOLagrangian(PPO):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Custom agent

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        _cfg = copy.deepcopy(PPOL_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)
        # =======================================================================
        # - get and process models from `self.models`
        # - populate `self.checkpoint_modules` dictionary for storing checkpoints
        # - parse configurations from `self.cfg`
        # - setup optimizers and learning rate scheduler
        # - set up preprocessors
        # =======================================================================
        # get cost value function
        self.cost_value = self.models.get("cost_value", None)

        # additional attributes
        self._cost_value_preprocessor = self.cfg["cost_value_preprocessor"]
        self._use_lagrangian = self.cfg["use_lagrangian"]
        self._lagrangian_pid = self.cfg["lagrangian_pid"]
        self._rescaling = self.cfg["rescaling"]
        self._cost_limit = self.cfg["cost_limit"]
        self._state_dependence = self.cfg["state_dependence"]
        self._mini_batches = self.cfg["mini_batches"]
        self._large_scale = self.cfg["large_scale"]

        # lagrangian
        if self._use_lagrangian:
            if not self._state_dependence:
                self._lagrangian_pid[1] = 0
                self._lagrangian_pid[2] = 0
                self.lag_optim = LagrangianOptimizer(self._lagrangian_pid)
            else:
                self.lag_optim = [LagrangianOptimizer(self._lagrangian_pid) for _ in range(self._mini_batches)]
        else:
            self.lag_optim = []

        # optimizer
        if self.cost_value is not None:
            self.optimizer = torch.optim.Adam(
                itertools.chain(
                    self.policy.parameters(), 
                    self.value.parameters(),
                    self.cost_value.parameters()), 
                lr=self._learning_rate
            )
        # cost value preprocessor
        if self._cost_value_preprocessor:
            self._cost_value_preprocessor = self._cost_value_preprocessor(**self.cfg["cost_value_preprocessor_kwargs"])
            self.checkpoint_modules["cost_value_preprocessor"] = self._cost_value_preprocessor
        else:
            self._cost_value_preprocessor = self._empty_preprocessor

        
        
    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        # =================================================================
        # - create tensors in memory if required
        # - # create temporary variables needed for storage and computation
        # =================================================================
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="costs", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="cost_advantages", size=1, dtype=torch.float32)
            
            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", 
                                   "values", "returns", "advantages", 
                                   "cost_values", "cost_returns", "cost_advantages"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # ======================================
        # - sample random actions if required or
        #   sample and return agent's actions
        # ======================================
        return super().act(states, timestep, timesteps)

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super(PPO, self).record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)


        # ========================================
        # - record agent's specific data in memory
        # ========================================
        if self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            costs = infos['cost'].reshape((-1, 1))
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)
                # TODO: add cost
                # costs = self._rewards_shaper(infos['cost'], timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)
                # TODO: compute cost values
                cost_values, _, _ = self.cost_value.act({"states": self._state_preprocessor(states)}, role="cost_value")
                cost_values = self._cost_value_preprocessor(cost_values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated
                # TODO: cost
                costs += self._discount_factor * cost_values * truncated

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                cost_values=cost_values,
                costs=costs,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    cost_values=cost_values,
                    costs=costs,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # =====================================
        # - call `self.update(...)` if required
        # =====================================
        # call parent's method for checkpointing and TensorBoard writing
        super().post_interaction(timestep, timesteps)

    def safety_loss(self, cost_value) -> Tuple[torch.tensor, dict]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lag = self.lag_optim.get_lag()
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1. / (lag + 1) if self._rescaling else 1
        stats = {"loss/rescaling": rescaling}
        loss = torch.mean(cost_value * lag)
        stats["loss/lagrangian"] = lag
        stats["loss/actor_safety"] = loss.item()
        return loss, stats
    
    def safety_loss_state_dep(self, cost_value) -> Tuple[torch.tensor, dict, torch.tensor]:
        """Compute the safety loss based on Lagrangian and return the scaling factor.

        :param list values: the cost values that want to be constrained. They will be
            multiplied with the Lagrangian multipliers.
        :return tuple[torch.tensor, dict]: the total safety loss and a dictionary of info
            (including the rescaling factor, lagrangian, safety loss etc.)
        """
        # get a list of lagrangian multiplier
        lag = self.cur_lag
        # Alg. 1 of http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf
        rescaling = 1. / (lag + 1)
        stats = {"loss/rescaling": torch.mean(rescaling).item()}
        loss = torch.mean(cost_value * lag)
        stats["loss/lagrangian"] = torch.mean(lag).item()
        stats["loss/actor_safety"] = loss.item()
        return loss, stats, rescaling
    
    def update_lagrangian(self, cost_value) -> None:
        """Update the Lagrangian multiplier before updating the policy.

        :param Union[List, float] cost_values: the estimation of cost values that want to
            be controlled under the target thresholds. It could be a list (multiple
            constraints) or a scalar value.
        """
        # if self._state_dependence:
        #     for i in range(self._mini_batches):
        #         self.lag_optim[i].step(cost_value[i], self._cost_limit)
        # else:
        self.lag_optim.step(cost_value, self._cost_limit)

    def update_lagrangian_per_state(self, cost_value) -> None:
        """Update the Lagrangian multiplier before updating the policy.

        :param Union[List, float] cost_values: the estimation of cost values that want to
            be controlled under the target thresholds. It could be a list (multiple
            constraints) or a scalar value.
        """
        # if self._state_dependence:
        #     for i in range(self._mini_batches):
        #         self.lag_optim[i].step(cost_value[i], self._cost_limit)
        # else:
        proj_cost_value = torch.clamp(cost_value - self._cost_limit, min=0)
        with torch.no_grad():
            self.cur_lag = self._lagrangian_pid[0] * proj_cost_value * self._large_scale

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # ===================================================
        # - implement algorithm's update step
        # - record tracking data using `self.track_data(...)`
        # ===================================================
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            self.cost_value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            last_cost_values, _, _ = self.cost_value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="cost_value"
            )
            self.value.train(True)
            self.cost_value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)
            last_cost_values = self._cost_value_preprocessor(last_cost_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        # compute cost returns and advantages
        cost_values = self.memory.get_tensor_by_name("cost_values")
        cost_returns, cost_advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("costs"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=cost_values,
            next_values=last_cost_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
        self.memory.set_tensor_by_name("cost_values", self._cost_value_preprocessor(cost_values, train=True))
        self.memory.set_tensor_by_name("cost_returns", self._cost_value_preprocessor(cost_returns, train=True))
        self.memory.set_tensor_by_name("cost_advantages", cost_advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
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
                
                # TODO: first update lagrangian multiplier
                if self._state_dependence:
                    self.update_lagrangian_per_state(sampled_cost_values)
                else:
                    self.update_lagrangian(cost_values)

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # TODO: safe policy update
                    # compute safety loss
                    if self._state_dependence:
                        loss_actor_safety, stats_actor, rescaling = self.safety_loss_state_dep(sampled_cost_advantages)
                    else:
                        loss_actor_safety, stats_actor = self.safety_loss(ratio * sampled_cost_advantages)
                    rescaling = stats_actor["loss/rescaling"]
                    policy_loss = rescaling * (policy_loss + loss_actor_safety)

                    # compute cost value loss
                    predicted_cost_values, _, _ = self.cost_value.act({"states": sampled_states}, role="cost_value")
                    cost_value_loss = self._value_loss_scale * F.mse_loss(sampled_cost_returns, predicted_cost_values)


                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + cost_value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data('Loss / Cost value loss', cost_value_loss.item())
        self.track_data("Loss / lag", stats_actor["loss/lagrangian"])
        self.track_data("Loss / policy safety loss", stats_actor["loss/actor_safety"])

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard

        Currently only scalar data are supported

        :param tag: Data identifier (e.g. 'Loss / policy loss')
        :type tag: str
        :param value: Value to track
        :type value: float
        """
        super().track_data(tag, value)
        wandb.log({tag: value})