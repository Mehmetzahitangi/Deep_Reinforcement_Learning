import torch
import torch.nn as nn
import numpy as np

# DeepMind First DQN Architecture
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__() # Get backpropagation and weight updates, automatically

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions) # No ReLU or Softmax. Not a probabilistic network, outputs are Q values. From -inf to +inf
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size())) # Product output dim (64*7*7)

    def forward(self, x):
        # Gelen uint8 matrisi float'a çevirip 0-1 arasına sıkıştırıyoruz
        normalized_x = x.float() / 255.0
        conv_out = self.conv(normalized_x).view(normalized_x.size()[0], -1) # flattening. x.size()[0] protect batch_size
        return self.fc(conv_out)