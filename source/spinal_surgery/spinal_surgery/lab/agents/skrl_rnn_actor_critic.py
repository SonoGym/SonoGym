import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, 
                 max_log_std=2, reduction="sum", num_envs=1, num_layers=1, hidden_size=128, sequence_length=16):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.features_extractor = nn.Sequential(nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                )
        self.net_features = nn.Linear(20160, 512)

        
        self.rnn = nn.GRU(input_size=512,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=False)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        
        self.net_value = nn.Sequential(nn.Linear(self.hidden_size, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        states = space

        features = self.features_extractor(states)

        features = self.net_features(features)

        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:, i0:i1, :], hidden_states)
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = features.view(-1, 1, features.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        if role == "policy":
            mean_actions = self.net(rnn_output)

            return mean_actions, self.log_std_parameter, {"rnn": [hidden_states]}
        
        elif role == "value":
            values = self.net_value(rnn_output)
            return values, {"rnn": [hidden_states]}   
