import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# Prioritized Replay Buffer Hiperparametreleri
BETA_START = 0.4
BETA_FRAMES = 100000

# distributional DQN Hiperparametreleri
Vmax = 10 # Oyunun verebileceği minimum ve maksimum değerleri tahmin edip sınırları çiziyoruz. Atari oyunları için genelde [-10, 10] yeterlidir
Vmin = -10
N_ATOMS = 51 # Kaç parça kullanacağız
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1) # 0.4 Atoms arasındaki mesafe -10.0, -9.6,....

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


class PrioReplayBuffer:

    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha # Alfa: Rastgelelik katsayısı (0.6)
        self.capacity = buf_size 
        self.pos = 0 # Buffer içinde veriyi nereye yazacağımızın indeksi
        self.buffer = [] # Gerçek tecrübelerin (state, action vb.) tutulduğu liste

        # Her tecrübenin öncelik puanını (hata payını, ne kadar önemli olduğunu) tutan numpy dizisi, buffer ile aynı boyutta.
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)
        self.beta = BETA_START # Beta: Önyargı düzeltme katsayısı (0.4)

    def update_beta(self, idx):
        # Beta değerini 0.4'ten başlatıp, adım adım 1.0'a kadar çıkaran fonksiyon. Her eğitim adımında çağrılır
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta
    
    def __len__(self):
        return len(self.buffer)
    
    def populate(self, count):
        # Torbadaki en yüksek öncelik değerini bul. Torba boşsa 1.0 verilir
        max_prio = self.priorities.max() if self.buffer else 1.0

        for _ in range(count):
            sample = next(self.exp_source_iter) # Oyundan yeni 1 frame (tecrübe) al

            # Buffer dolana kadar ekle, dolduysa en eskisinin üzerine yaz (pos)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample

            # Yeni gelen her veriye torbadaki EN YÜKSEK önceliği verir. Bu sayede, yeni verinin eğitimde kesinlikle en az bir kere seçilip ağdan geçmesi garantilenir
            self.priorities[self.pos] = max_prio 
            # Yazma işaretçisini bir sonrakine kaydırır (Döngüsel/Circular buffer mantığı)
            self.pos = (self.pos + 1) % self.capacity

    # Öncelikli Veri Çekme
    def sample(self, batch_size):
        # Buffer tam dolu değilse, sadece dolu olan kısma kadar olan öncelikleri al
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # P(i) = p_i^α formülünü uygula (Alfa ile güç ayarlama)
        probs = prios ** self.prob_alpha

        # Olasılıkları normalize et (Hepsini toplayıp 1'e bölüyoruz, %100 üzerinden değerler çıksın)
        probs /= probs.sum()
        
        # Verilen olasılıklara (p=probs) göre seçim yap
        indices = np.random.choice(len(self.buffer), batch_size, p=probs) # p=probs ==> hata payı yüksek olanların seçilme ihtimali çok daha yüksek
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)

        # w_i = (N * P(i))^-β formülünün uygulanması (Önyargı düzeltme ağırlıkları)
        weights = (total * probs[indices]) ** (-self.beta)

        # Ağırlıkların normalize edilmesi (En yükseği 1 olacak şekilde oranlıyoruz, eğitimin çökmesini engellemek için)
        weights /= weights.max()

        # Seçimleri (samples), onların buffer'daki indekslerini ve ağırlıklarını dön
        return samples, indices, np.array(weights, dtype=np.float32)
    
    # Hafıza Güncelleme
    def update_priorities(self, batch_indices,  batch_priorities):
        """
        Ajan 32'li veriyi çekti, sinir ağından (calc_loss) geçirdi ve "O kadar da hatalı olmadı, 10 hata beklerken 2 hata yaptı". O yeni hata değerleri bu fonskiyona gönderilir
        Bu sayede ajan öğrendikçe, eskiden zorlandığı anıların öncelik puanı düşer ve o anıları bir daha kolay kolay karşısına çıkarmaz. Sistemi sürekli canlı ve "en zor" olana odaklı tutar
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            # Seçilen tecrübelerin eski önceliklerini, ağdan dönen YENİ HATA (Loss) değerleriyle değiştir
            self.priorities[idx] = prio

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Avantaj Akışı, Şu aksiyonu yaparsam ortalamadan ne kadar daha iyi olur?
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Durum Değeri Akışı, Şu an bulunulan durum genel olarak ne kadar iyi/kötü?
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)

        # Q(s,a) = V(s) + ( A(s,a) - mean(A) )
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256 # Görüntüyü float'a çevirir ve normalize eder (RAM tasarrufu)
        conv_out = self.conv(fx).view(fx.size()[0], -1) # # Pikselleri düz uzun bir vektör yapar

        # Aynı görüntüyü hem Avantaj hem de Değer kısmına gönder ve sonuçları getir
        return self.fc_adv(conv_out), self.fc_val(conv_out)
    

class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

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
            nn.Linear(512, n_actions * N_ATOMS) # n_actions ==> n_actions * N_ATOMS
        )

        sups = torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
        self.register_buffer("supports", sups)
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())
    
def distr_projection(next_distr, rewards, dones, gamma):
    """
    Distribution Projection (Categorical Algorithm) uygulanması, "A Distributional Perspective on RL" Makalesinden
    Bu fonksiyon, Categorical DQN'in hedef dağılımını hesaplayan kalbidir. Bu kod olmazsa ağın çizdiği histogramlar sürekli kayar ve sistem çöker
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS),  dtype=np.float32) # Sıfırdan boş bir hedef dağılım (proj_distr) matrisi yaratıyoruz

    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    for atom in range(N_ATOMS):
        # KAYDIRMA: Bellman Update 
        v = rewards + (Vmin + atom * delta_z) * gamma
        # SINIRLAMA: CLIPPING
        tz_j = np.minimum(Vmax, np.maximum(Vmin, v))

        b_j = (tz_j - Vmin) / delta_z # Yeni değerin atom indekslerinde nereye denk geldiğini bulur
        l = np.floor(b_j).astype(np.int64) # Alt Atom İndeksi
        u = np.ceil(b_j).astype(np.int64) # Üst Atom İndeksi

        # Interpolasyon
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]


    if dones.any():
        proj_distr[dones] = 0.0 # Biten oyunların geleceğini sıfırla (çöpe at)
        tz_j = np.minimum(
            Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = \
                (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = \
                (b_j - l)[ne_mask]
    return proj_distr


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        Dueling Ve Noisy Network kısmı burada
        """
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Advantage
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            NoisyFactorizedLinear(conv_out_size, 256), # Gürültülü Ağ
            nn.ReLU(),
            NoisyFactorizedLinear(256, n_actions) # Gürültülü Ağ, n_actions var n_actions * N_ATOMS değil
        )

        # State Value
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256), # Standart Ağ
            nn.ReLU(),
            nn.Linear(256, 1) # Standart Ağ
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        # Q(s,a) = V(s) + ( A(s,a) - ortalama(A) )
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)