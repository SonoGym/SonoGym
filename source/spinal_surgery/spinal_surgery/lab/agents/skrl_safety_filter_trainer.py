from __future__ import annotations

import dataclasses
import sys
from typing import List, Optional, Union

import torch
import tqdm

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.utils import ScopedTimer


@dataclasses.dataclass(kw_only=True)
class SafetyFilterSequentialTrainerCfg(SequentialTrainerCfg):
    """Configuration for the safety-filter sequential trainer (alias of the base cfg)."""


# Backwards-compatible default-config dict.
SEQUENTIAL_TRAINER_DEFAULT_CONFIG: dict = {
    "timesteps": 100000,
    "headless": False,
    "render_interval": 1,
    "disable_progressbar": False,
    "close_environment_at_exit": True,
    "environment_info": "episode",
    "stochastic_evaluation": False,
}


class SafetyFilterSequentialTrainer(Trainer):
    """Sequential trainer compatible with skrl 2.x.

    For a single agent the parent ``Trainer.train`` / ``Trainer.eval`` already implement
    the loop we need; this class delegates to them. The simultaneous-agent path is
    re-implemented to keep the previous behavior.
    """

    def __init__(
        self,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, MultiAgent, List[Agent], List[MultiAgent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[Union[SafetyFilterSequentialTrainerCfg, dict]] = None,
    ) -> None:
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            # filter to only fields known by the dataclass
            valid_keys = {f.name for f in dataclasses.fields(SafetyFilterSequentialTrainerCfg)}
            trainer_cfg = SafetyFilterSequentialTrainerCfg(**{k: v for k, v in cfg.items() if k in valid_keys})
        else:
            trainer_cfg = cfg

        super().__init__(
            env=env,
            agents=agents,
            scopes=agents_scope if agents_scope is not None else [],
            cfg=trainer_cfg,
        )

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        # set training mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(True)
        else:
            self.agents.enable_training_mode(True)

        # single agent: defer to the base trainer's loop (works for vec envs)
        if self.num_simultaneous_agents == 1:
            super().train()
            return

        # simultaneous agents
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout
        ):
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    _actions.append(actions)
                    _outputs.append(outputs)
                actions = torch.vstack(_actions)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )

                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

    def eval(self) -> None:
        # set evaluation mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.enable_training_mode(False)
        else:
            self.agents.enable_training_mode(False)

        # single agent: defer to the base trainer's loop
        if self.num_simultaneous_agents == 1:
            super().eval()
            return

        # simultaneous agents
        observations, infos = self.env.reset()
        states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.cfg.timesteps), disable=self.cfg.disable_progressbar, file=sys.stdout
        ):
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            with torch.no_grad():
                _actions, _outputs = [], []
                for agent, scope in zip(self.agents, self.scopes):
                    with ScopedTimer() as timer:
                        actions, outputs = agent.act(
                            observations[scope[0] : scope[1]],
                            states[scope[0] : scope[1]] if states is not None else None,
                            timestep=timestep,
                            timesteps=self.cfg.timesteps,
                        )
                        agent.track_data("Stats / Inference time (ms)", timer.elapsed_time_ms)
                    chosen = actions if self.cfg.stochastic_evaluation else outputs.get("mean_actions", actions)
                    _actions.append(chosen)
                    _outputs.append(outputs)
                actions = torch.vstack(_actions)

                with ScopedTimer() as timer:
                    next_observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    next_states = self.env.state()
                    elapsed_time_ms = timer.elapsed_time_ms
                    for agent in self.agents:
                        agent.track_data("Stats / Env stepping time (ms)", elapsed_time_ms)

                if not self.cfg.headless and not timestep % self.cfg.render_interval:
                    self.env.render()

                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]] if states is not None else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]] if next_states is not None else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.cfg.timesteps,
                    )

                if self.cfg.environment_info in infos:
                    for k, v in infos[self.cfg.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(k if "/" in k else f"Info / {k}", v.item())

            # eval: do checkpointing/logging side-effects only (skip the algorithm update)
            for agent in self.agents:
                super(agent.__class__, agent).post_interaction(timestep=timestep, timesteps=self.cfg.timesteps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        observations, infos = self.env.reset()
                        states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states
