import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.features_extractor = nn.Sequential(nn.Conv2d(25, 32, kernel_size=8, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(2048, 512)

        self.net_pos = nn.Sequential(nn.Linear(3, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 64))
        self.net_quat = nn.Sequential(nn.Linear(4, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 64))

        self.net = nn.Sequential(nn.Linear(512 + 64 + 64, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        image = space['image']
        pos = space['pos']
        quat = space['quat']

        features = self.net_features(self.features_extractor(image))
        pos_features = self.net_pos(pos)
        quat_features = self.net_quat(quat)

        mean_actions = self.net(torch.cat([features,
                                          pos_features,
                                          quat_features], dim=-1))

        return mean_actions, self.log_std_parameter, {}
    

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(25, 32, kernel_size=8, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(2048, 512)

        self.net_pos = nn.Sequential(nn.Linear(3, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 64))
        self.net_quat = nn.Sequential(nn.Linear(4, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 64))

        self.net = nn.Sequential(nn.Linear(512 + 64 + 64, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        image = space['image']
        pos = space['pos']
        quat = space['quat']

        features = self.net_features(self.features_extractor(image))
        pos_features = self.net_pos(pos)
        quat_features = self.net_quat(quat)

        values = self.net(torch.cat([features,
                                     pos_features,
                                     quat_features], dim=-1))

        return values, {}
    



    

class QNet(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(observation_space['image'].shape[0], 32, kernel_size=8, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(2048, 512)

        self.net_actions = nn.Sequential(nn.Linear(6, 128),
                                        nn.ELU(),
                                        nn.Linear(128, 128),
                                        nn.ELU(),
                                        nn.Linear(128, 64))

        self.net_pos = nn.Sequential(nn.Linear(3, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 64))
        self.net_quat = nn.Sequential(nn.Linear(4, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 64))

        self.net = nn.Sequential(nn.Linear(512 + 64 + 64 + 64, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        states = inputs["states"]
        actions = inputs["taken_actions"]
        space = self.tensor_to_space(states, self.observation_space)

        image = space['image']
        pos = space['pos']
        quat = space['quat']

        features = self.net_features(self.features_extractor(image))
        pos_features = self.net_pos(pos)
        quat_features = self.net_quat(quat)
        action_features = self.net_actions(actions)

        values = self.net(torch.cat([features,
                                     pos_features,
                                     quat_features,
                                     action_features], dim=-1))

        return values, {}
    


class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.features_extractor = nn.Sequential(nn.Conv2d(observation_space['image'].shape[0], 32, kernel_size=8, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(2048, 512)

        self.net_pos = nn.Sequential(nn.Linear(3, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 128),
                                     nn.ELU(),
                                     nn.Linear(128, 64))
        self.net_quat = nn.Sequential(nn.Linear(4, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 128),
                                      nn.ELU(),
                                      nn.Linear(128, 64))

        self.net = nn.Sequential(nn.Linear(512 + 64 + 64, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        
        self.net_value = nn.Sequential(nn.Linear(512 + 64 + 64, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        image = space['image']
        pos = space['pos']
        quat = space['quat']

        features = self.net_features(self.features_extractor(image))
        pos_features = self.net_pos(pos)
        quat_features = self.net_quat(quat)

        if role == "policy":
            mean_actions = self.net(torch.cat([features,
                                                pos_features,
                                                quat_features], dim=-1))

            return mean_actions, self.log_std_parameter, {}
        elif role == "value":
            values = self.net_value(torch.cat([features,
                                        pos_features,
                                        quat_features], dim=-1))
            return values, {}   


class SharedModelSAC(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.features_extractor = nn.Sequential(nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(20160, 512)
        
        self.net_actions = nn.Sequential(nn.Linear(action_space.shape[0], 128),
                                         nn.ELU(),
                                         nn.Linear(128, 128),
                                         nn.ELU(),
                                         nn.Linear(128, 64))

        self.net = nn.Sequential(nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        
        self.net_value = nn.Sequential(nn.Linear(512 + 64, 256),
                                       nn.ELU(),
                                       nn.Linear(256, 128),
                                       nn.ELU(),
                                       nn.Linear(128, 1))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "critic_1" or role == "critic_2" or role == "target_critic_1" or role == "target_critic_2":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)

        image = space

        features = self.net_features(self.features_extractor(image))
    
        if role == "policy":
            mean_actions = self.net(features)

            return mean_actions, self.log_std_parameter, {}
        elif role == "critic_1" or role == "critic_2" or role == "target_critic_1" or role == "target_critic_2":
            actions = inputs["taken_actions"]
            action_features = self.net_actions(actions)
            values = self.net_value(torch.cat([features, action_features], dim=-1))
            return values, {}   