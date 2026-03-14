import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F




class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017 ,bias = True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w) # sigma_weight ve sigma_bias: Gürültünün şiddetidir. Ağın ne kadar çok keşif yapması gerektiğini belirler. nn.Parameter olarak tanımlanır, yani ağ oyunu oynarken geriye yayılım (backpropagation) ile bu gürültü miktarını da öğrenip günceller
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z) # Gürültünün kendisidir. Bu rastgeledir, öğrenilmez. register_buffer, PyTorch'a "Bu sadece sabit bir matris, bunu ekran kartına (GPU) taşı ama üzerinde eğitim (gradient) yapma" demektir
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w) 
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input): 
        self.epsilon_weight.normal_() # Matrisin içi rastgele sayılarla (çan eğrisi) doldurulur.
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data +  self.weight # Standart ağırlıklara, şiddeti ayarlanmış rastgele gürültü eklenir
        return F.linear(input, v, bias) # Input verisi bu yeni "titreyen ağırlıklarla" çarpılıp bir sonraki katmana iletilir


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    Hızlı olan versiyon
    """
    def __init__(self, in_features, out_features,
                 sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1) #  epsilon_weight gibi büyük bir matris yerine sadece epsilon_input ve epsilon_output var
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out) # Çok küçük olan eps_in ve eps_out vektörleri birbiriyle çapraz çarpılarak, asıl ağırlık matrisi boyutunda devasa gürültü matrisi (noise_v) saniyenin çok küçük bir diliminde elde edilir. Kalan işlemler bağımsız yöntemle birebir aynıdır
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)
    
class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyFactorizedLinear(conv_out_size, 512),
            NoisyFactorizedLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self): 
        """
        SNR (Signal-to-Noise Ratio)

        Sinyal (layer.weight): Ağın oyunu oynayarak öğrendiği asıl bilgi, net zekasıdır
        Gürültü (layer.sigma_weight): Ağın keşfetmek için kullandığı titreşim, kararsızlığıdır
        """
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]
