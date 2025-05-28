import torch
import torch.nn as nn

# ---------------------------- Utility classes ---------------------------------
class MLP(nn.Module):
    def __init__(self, poly_dim, feat_dim=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(poly_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, feat_dim), nn.ReLU(),
        )
    def forward(self, poly):
        poly = poly.view(poly.size(0), -1)
        return self.body(poly)

class CNN1Branch(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((8, 8))
        )
        self.out_dim = out_channels * 8 * 8

    def forward(self, mask):
        f = self.feat(mask)
        return f.view(f.size(0), -1) # flatten

class CNN2Branch(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((8, 8))
        )
        self.out_dim = out_channels * 8 * 8

    def forward(self, img):
        f = self.feat(img)
        return f.view(f.size(0), -1)

# --------------------------------- Models -------------------------------------
class MLPModel(nn.Module):
    def __init__(self, poly_dim, mlp_feat_dim=32):
        super().__init__()
        self.mlp = MLP(poly_dim, mlp_feat_dim)
        self.regressor = nn.Sequential(
            nn.Linear(mlp_feat_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 2)
        )

    def forward(self, poly):
        feat = self.mlp(poly)
        return self.regressor(feat)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNN1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = CNN1Branch()
        self.regressor = nn.Sequential(
            nn.Linear(self.cnn1.out_dim, 128), # (8192, 128)
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, mask):
        feat = self.cnn1(mask)
        return self.regressor(feat)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNN2Model(nn.Module):
    def __init__(self, out_dim=128*8*8):
        super().__init__()
        self.cnn2 = CNN2Branch(out_dim)
        self.regressor = nn.Sequential(
            nn.Linear(self.cnn2.out_dim, 128), # (8192, 128)
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, img):
        feat = self.cnn2(img)
        return self.regressor(feat)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLP_CNN1_Model(nn.Module):
    def __init__(self, poly_dim, mlp_feat_dim=32):
        super().__init__()
        self.mlp = MLP(poly_dim, mlp_feat_dim)
        self.cnn1 = CNN1Branch()
        self.fuse = nn.Sequential(
            nn.Linear(mlp_feat_dim + self.cnn1.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, mask, poly):
        feat1 = self.mlp(poly)
        feat2 = self.cnn1(mask)
        fused = torch.cat([feat1, feat2], dim=1)
        return self.fuse(fused)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLP_CNN2_Model(nn.Module):
    def __init__(self, poly_dim, mlp_feat_dim=32):
        super().__init__()
        self.mlp = MLP(poly_dim, mlp_feat_dim)
        self.cnn2 = CNN2Branch()
        self.fuse = nn.Sequential(
            nn.Linear(mlp_feat_dim + self.cnn2.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, img, poly):
        feat1 = self.mlp(poly)
        feat2 = self.cnn2(img)
        fused = torch.cat([feat1, feat2], dim=1)
        return self.fuse(fused)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNN1_CNN2_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = CNN1Branch()
        self.cnn2 = CNN2Branch()
        self.fuse = nn.Sequential(
            nn.Linear(self.cnn1.out_dim + self.cnn2.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, mask, img):
        feat1 = self.cnn1(mask)
        feat2 = self.cnn2(img)
        fused = torch.cat([feat1, feat2], dim=1)
        return self.fuse(fused)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLP_CNN1_CNN2_Model(nn.Module):
    def __init__(self, poly_dim, mlp_feat_dim=32):
        super().__init__()
        self.mlp = MLP(poly_dim, mlp_feat_dim)
        self.cnn1 = CNN1Branch()
        self.cnn2 = CNN2Branch()
        self.fuse = nn.Sequential(
            nn.Linear(mlp_feat_dim + self.cnn1.out_dim + self.cnn2.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, mask, img, poly):
        feat1 = self.mlp(poly)
        feat2 = self.cnn1(mask)
        feat3 = self.cnn2(img)
        fused = torch.cat([feat1, feat2, feat3], dim=1)
        return self.fuse(fused)

    def count_parameters(self):
      return sum(p.numel() for p in self.parameters() if p.requires_grad)