import torch
from torch import nn

class MultiInputNN(nn.Module):
    def __init__(self, input_shape, output_dim=128):
        super(MultiInputNN, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layer for image features
        self.fc_image = nn.Linear(1536, 512)
        
        # FNN for pos
        self.fc_pos = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # FNN for quat
        self.fc_quat = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # Final FNN after concatenation
        self.fc_final = nn.Sequential(
            nn.Linear(512 + 256 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.device = None
        
    def forward(self, x, state=None):
        if self.device is None:
            self.device = next(self.parameters()).device
        image = torch.tensor(x['image']).to(torch.float32).to(self.device)
        pos = torch.tensor(x['pos']).to(torch.float32).to(self.device)
        quat = torch.tensor(x['quat']).to(torch.float32).to(self.device)
        # image = image[0,...]
        # pos = pos[0,...]
        # quat = quat[0,...]
        
        # Process image
        img_features = self.cnn(image)
        img_features = torch.flatten(img_features, start_dim=1)
        # if self.fc_image is None:
        #     self.fc_image = nn.Linear(img_features.shape[1], 512).to(img_features.device)
        #     print('img_feature_shape', img_features.shape[1])
        img_features = self.fc_image(img_features)
        
        # Process pos and quat
        pos_features = self.fc_pos(pos)
        quat_features = self.fc_quat(quat)
        
        # Concatenate all features
        combined = torch.cat([img_features, pos_features, quat_features], dim=1)
        
        # Final processing
        output = self.fc_final(combined)
        # output = output.unsqueeze(0)
        
        return output, state
    

class MultiInputActionNN(nn.Module):
    def __init__(self, action_dim, output_dim):
        super(MultiInputNN, self).__init__()
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layer for image features
        self.fc_image = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),  # Assuming final feature map size is 6x6
            nn.ReLU()
        )
        
        # FNN for pos
        self.fc_pos = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # FNN for quat
        self.fc_quat = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
         # FNN for quat
        self.fc_action = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Final FNN after concatenation
        self.fc_final = nn.Sequential(
            nn.Linear(512 + 256 + 256 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x, action, state=None):
        image, pos, quat = x['image'], x['pos'], x['quat']
        
        # Process image
        img_features = self.cnn(image)
        img_features = torch.flatten(img_features, start_dim=1)
        img_features = self.fc_image(img_features)
        
        # Process pos and quat
        pos_features = self.fc_pos(pos)
        quat_features = self.fc_quat(quat)
        act_features = self.fc_action(action)
        
        # Concatenate all features
        combined = torch.cat([img_features, pos_features, quat_features, act_features], dim=1)
        
        # Final processing
        output = self.fc_final(combined)
        
        return output, state