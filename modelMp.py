import torch
import torch.nn as nn

class convblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class binmnistmodel(nn.Module):
    def __init__(self):
        super(binmnistmodel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            convblock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            convblock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        '''self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            convblock(32, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )'''
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float()
        encoded = self.encoder(x)
        fc_out = self.fc_layers(encoded)
        decoded = self.decoder(fc_out)
        #print(decoded.shape)
        return decoded


def make_noise_sample_p(image, t=10, include_last = True, include_first = True):
    li = []
    R = torch.randint(2, size=image.shape, dtype=torch.bool)
    if include_last:
        noisy_image = torch.logical_xor(image, R)
        li.append(noisy_image.clone()) 
    ctm = None
    for _ in range(t):
        c = torch.nonzero(R, as_tuple=True)
        ct = c[0].shape[0]
        if not ctm:
            ctm = ct
        if not include_first and ctm//t + 1 > ct:
            break
        noise_index = torch.randperm(ct)[:min(ct, ctm//t + 1)] 
        R[0, c[1][noise_index], c[2][noise_index]] = 0
        
        noisy_image = torch.logical_xor(image, R)
        li.append(noisy_image.clone()) 
    li.reverse()
    return li

def make_noise_sample(image,tot,t,M,R):
    B = R.clone()
    B[M<(1-t/tot)] = 0
    return torch.logical_xor(image, B)

def make_noise_sample_s(image, t=10, include_last = True, include_first = True):
    li = []
    M = torch.rand(size=image.shape)
    #M = M**2
    R = torch.randint(2, size=image.shape, dtype=torch.bool)
    if include_first:
        li.append(image)
    for n in range(1,t):
        noisy_image = make_noise_sample(image, t, n, M, R)
        li.append(noisy_image.clone()) 
    if include_last:
        li.append(torch.logical_xor(image, R))
    return li