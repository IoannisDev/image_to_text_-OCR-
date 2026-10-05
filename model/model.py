import math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor


#Dbnet Model 
# This model is responsible for  identifying a box boudary around text 
class DBNet(nn.Module):
    def __init__(self, k=50, pretrained=True, prob_only=False):
        super().__init__()
        self.prob_only = prob_only      # True for inference: skip the threshold head
        # pretrained=False for inference: the checkpoint overwrites these weights anyway
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.body = create_feature_extractor(
            r, {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"})
        in_chs = [64, 128, 256, 512]
        self.lateral = nn.ModuleList([nn.Conv2d(c, 256, 1, bias=False) for c in in_chs])
        self.smooth = nn.ModuleList([nn.Conv2d(256, 64, 3, padding=1, bias=False) for _ in in_chs])
        self.prob_head, self.thresh_head = self._head(), self._head()
        self.k = k

    @staticmethod
    def _head():
        return nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, stride=2))                     # logits, full resolution

    def forward(self, x):                                               # H, W divisible by 32
        f = self.body(x)
        lat = [l(f[n]) for l, n in zip(self.lateral, ["c2", "c3", "c4", "c5"])]
        for i in range(3, 0, -1):                                       # top-down pathway
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")
        size = lat[0].shape[-2:]
        fused = torch.cat([F.interpolate(s(l), size=size, mode="nearest")
                           for s, l in zip(self.smooth, lat)], dim=1)   # (B, 256, H/4, W/4)
        p_logit = self.prob_head(fused)                             # P only at inference
        if self.prob_only:
            return p_logit
        T = torch.sigmoid(self.thresh_head(fused))
        b_logit = self.k * (torch.sigmoid(p_logit) - T)                 # DB step, as logits
        return p_logit, T, b_logit


def balanced_bce(logit, gt, mask, neg_ratio=3.0, min_neg=64, eps=1e-6):
    logit = logit.float()                      # fp16 sums overflow
    pos = (gt > 0.5) & (mask > 0.5)
    neg = (gt <= 0.5) & (mask > 0.5)
    n_pos = int(pos.sum())
    n_neg = min(int(neg.sum()), max(int(n_pos * neg_ratio), min_neg))
    loss = F.binary_cross_entropy_with_logits(logit, gt, reduction="none")
    neg_loss = loss[neg].topk(n_neg).values.sum() if n_neg > 0 else loss.new_zeros(())
    return (loss[pos].sum() + neg_loss) / (n_pos + n_neg + eps)

#Full loss function that use DICE and BCE loss
def my_loss(raw, batch):
    p_logit, T, b_logit = (t.float() for t in raw)
    gt, mask = batch["prob_map"], batch["prob_mask"]

    l_p = balanced_bce(p_logit, gt, mask)

    # Dice, not BCE: b_logit spans +/-50 and BCE on it swamps L_P
    b = torch.sigmoid(b_logit)
    inter = (b * gt * mask).sum()
    l_b = 1.0 - (2 * inter + 1e-6) / ((b * mask).sum() + (gt * mask).sum() + 1e-6)

    l_t = (torch.abs(T - batch["thresh_map"]) * batch["thresh_mask"]).sum() \
          / (batch["thresh_mask"].sum() + 1e-6)

    return l_p + l_b + 10.0 * l_t, {"L_P": l_p.item(), "L_B": l_b.item(), "L_T": l_t.item()}


def db_loss(p_logit, T, b_logit, gt_shrink, mask, gt_thresh, thresh_mask, alpha=1.0, beta=10.0):
    l_p = balanced_bce(p_logit, gt_shrink, mask)
    l_b = balanced_bce(b_logit, gt_shrink, mask)
    l_t = (torch.abs(T - gt_thresh) * thresh_mask).sum() / (thresh_mask.sum() + 1e-6)
    return l_p + alpha * l_b + beta * l_t

#CNN 
class CNN_FE(nn.Module):
  def __init__(self) :
      super().__init__()
      self.block1 = nn.Sequential(
          nn.Conv2d(1,64,kernel_size=3,padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d((2,2),(2,2)))
      self.block2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,2),(2,2))
                                  )
      self.block3 = nn.Sequential(
          nn.Conv2d(128,256,kernel_size=3,padding=1),
          nn.Conv2d(256,256,kernel_size=3,padding =1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.MaxPool2d((2,1),(2,1))
      )
      self.block4 = nn.Sequential(
          nn.Conv2d(256,512,kernel_size=3,padding=1),
          nn.Conv2d(512,512,kernel_size=3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d((2,1),(2,1))
          )
      self.block5 = nn.Sequential(
          nn.Conv2d(512,512,kernel_size=3,padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.MaxPool2d((2,1),(2,1))
      )
  def forward(self,X):
    Z = self.block1(X)
    Z = self.block2(Z)
    Z = self.block3(Z)
    Z = self.block4(Z)
    out = self.block5(Z)
    return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))     

    def forward(self, x):                                
        return x + self.pe[:x.size(0)]


class TransformerEncoderHead(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8,
                 num_layers=3, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
            activation="gelu",
            norm_first=True)          # pre-LN: trains without heavy warmup
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):                   
        x = x.squeeze(2).permute(2, 0, 1)                 
        x = self.drop(self.pos(x))

        mask = None
        if lengths is not None:
            T = x.size(0)
            idx = torch.arange(T, device=x.device)[None, :]
            mask = idx >= lengths.to(x.device)[:, None]  
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.fc(x).log_softmax(dim=2)

class BILstm(nn.Module):
  def __init__(self,num_classes) :
      super().__init__()
      self.Bi_lstm  = nn.LSTM(512,hidden_size=256,num_layers=2,bidirectional = True)
      self.Ln = nn.Linear(512,num_classes)
  def forward(self,X):
    B,W,H,T = X.shape
    X = torch.squeeze(X,dim=2)
    X = X.permute(2,0,1)
    out,_ = self.Bi_lstm(X)
    out = self.Ln(out)
    return torch.log_softmax(out,dim=2)

# This is a past iteration of the model that
# used Bidirectional LSTM , instead of this 
# transformer was used as  replacement

class crnn_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn_layer = CNN_FE()
    self.bi_lstm = BILstm()
  def forward(self,x):
    out = self.cnn_layer(x)
    out = self.bi_lstm(x)
    return out