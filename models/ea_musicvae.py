# Music Generation Model
import torch
import torch.nn as nn

class FlowCoupling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, dim))
        self.translate = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, dim))

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        s = self.scale(z1)
        t = self.translate(z1)
        z2 = z2 * torch.exp(s) + t
        return torch.cat([z1, z2], dim=1)

class EA_MusicVAE(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.flow = nn.Sequential(*[FlowCoupling(hidden_dim) for _ in range(4)])
        self.decoder = nn.GRU(hidden_dim + 256, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, emotion_embedding):
        _, (h, _) = self.encoder(x)
        z = self.flow(h[-1])
        z = torch.cat([z, emotion_embedding], dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(z)
        return self.output(out)