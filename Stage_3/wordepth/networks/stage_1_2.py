import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Stage_2_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps_net=nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,256)
        )
    def forward(self, image_feature):

        image_feature=image_feature.to(torch.float32)
        eps=self.eps_net(image_feature)

        return eps

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, prior_mean = 1.54):
        super(OutConv, self).__init__()

        self.prior_mean = prior_mean
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.exp(self.conv(x) + self.prior_mean)

class FourierPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        d_model = hidden_size // 2
        self.pe = torch.zeros(1, resolution[0], resolution[1], hidden_size)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pos_x = torch.arange(resolution[0]).unsqueeze(1)
        pe_x = torch.zeros(resolution[0], d_model)
        pe_x[:, 0::2] = torch.sin(pos_x * div_term)
        pe_x[:, 1::2] = torch.cos(pos_x * div_term)
        pe_x = pe_x[:,0:d_model]
        pe_x = pe_x[None, :, None, :].repeat(1, 1, resolution[1], 1)

        pos_y = torch.arange(resolution[1]).unsqueeze(1)
        pe_y = torch.zeros(resolution[1], d_model)
        pe_y[:, 0::2] = torch.sin(pos_y * div_term)
        pe_y[:, 1::2] = torch.cos(pos_y * div_term)
        pe_y = pe_y[:,0:d_model]
        pe_y = pe_y[None, None, :, :].repeat(1, resolution[0], 1, 1)
       
        self.pe = torch.cat((pe_x, pe_y), dim=3)

    def forward(self, inputs):
        pe=self.pe.to(inputs.device).repeat(inputs.shape[0],1,1,1).permute(0,3,2,1)
        return pe + inputs

class VarLayer(nn.Module):
    def __init__(self, in_channels, h, w):
        super(VarLayer, self).__init__()

        self.pos_0 = FourierPositionEmbed(in_channels, [4,4])
        self.conv_0=nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, padding_mode="replicate")
        self.relu_0=nn.LeakyReLU()


        self.pos_1 = FourierPositionEmbed(256, [8,8])
        self.conv_1=nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode="replicate")
        self.relu_1=nn.LeakyReLU()

        self.pos_2 = FourierPositionEmbed(256, [20,15])
        self.conv_2=nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode="replicate")
        self.relu_2=nn.LeakyReLU()



    def forward(self, room_feat):
        # text_feature,1,1024,1,1
        # Upsampling here, to 8*8
        # Or, FC Layer, 1*1024*1*1 -> 1*(1024*4)*1*1 -> 1*1024*2*2

        d=F.interpolate(room_feat, scale_factor=4, mode='bilinear', align_corners=True) # 4,4
        d=self.pos_0(d)
        d=self.conv_0(d) # 512
        d=self.relu_0(d) # elu, LeakyReLU (specify alpha)


        d=F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True) # 8,8
        d=self.pos_1(d)
        d=self.conv_1(d) # 512
        d=self.relu_1(d) # elu, LeakyReLU (specify alpha)

        

        d=F.interpolate(d, size=(15, 20), mode='bilinear', align_corners=False) # 15,20
        d=self.pos_2(d)
        d=self.conv_2(d) # 128
        d=self.relu_2(d)

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True) # 30,40

        return d

        # print(d.shape)，torch.Size([1, 128, 30, 40])
        



class Refine(nn.Module):
    def __init__(self, c1, c2):
        super(Refine, self).__init__()

        self.dw = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1))

    def forward(self, depth): # gradually upsample, depth feature dim=128, feat dim=512,256,64

        depth_new = self.dw(depth)

        return depth_new


class MetricLayer(nn.Module):
    def __init__(self, c):
        super(MetricLayer, self).__init__()

        self.ln = nn.Sequential(
                nn.Linear(c, c//4),
                nn.LeakyReLU(),
                nn.Linear(c//4, 2))

    def forward(self, x):

        x = self.ln(x)
        x = x.unsqueeze(-1).unsqueeze(-1)

        return x


class Stage_1_Model(nn.Module):
    def __init__(self, pretrained=None, max_depth=10.0, prior_mean=1.54, si_lambda=0.85, img_size=(480, 640),hidden_dim=256,std_reg=1):
        super().__init__()

        self.prior_mean = prior_mean
        self.SI_loss_lambda = si_lambda
        self.max_depth = max_depth

        

        self.outc = OutConv(128, 1, self.prior_mean)

        self.vlayer = VarLayer(hidden_dim, img_size[0]//16, img_size[1]//16)

        self.ref_4 = Refine(256, 128)
        self.ref_3 = Refine(128, 128)
        self.ref_2 = Refine(128, 128)


        self.mlayer = MetricLayer(hidden_dim)

        self.mean=nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,hidden_dim)
        )

        self.deviation=nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,hidden_dim)
        )


        self.std_reg=std_reg

    def forward(self, image, gts=None, text_feature=None,eps=None):
        inter_feature=[]
        if eps==None:
            raise()

        outs={}

        text_feature=text_feature.to(torch.float32)


        mean=self.mean(text_feature)
        logvar=self.deviation(text_feature)
        std = torch.exp(0.5 * logvar)
        
        # eps = torch.randn_like(std) # This is for stage 1 training

        room_feat = mean + eps * std
        std_norm=torch.sum(std)/(std.shape[0]*std.shape[1])


        
        metric = self.mlayer(room_feat) # torch.Size([1, 2, 1, 1])

        room_feat=room_feat.unsqueeze(-1).unsqueeze(-1) # 1,1024,1,1

        d = self.vlayer(room_feat) # torch.Size([1, 256, 30, 40])
        inter_feature.append(d) # 0

        d  = self.ref_4(d)
        inter_feature.append(d) # 1

        d_u4 = F.interpolate(d, scale_factor=16, mode='bilinear', align_corners=True)

        d = self.ref_3(F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))
        inter_feature.append(d) # 2

        d_u3 = F.interpolate(d, scale_factor=8, mode='bilinear', align_corners=True)

        d = self.ref_2(F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True))
        inter_feature.append(d) # 3

        d_u2 = F.interpolate(d, scale_factor=4, mode='bilinear', align_corners=True)

        d = d_u2 + d_u3 + d_u4 # torch.Size([1, 128, 480, 640])
        inter_feature.append(d) # 4
        
        d = torch.sigmoid(metric[:, 0:1]) * (self.outc(d) + torch.exp(metric[:, 1:2]))


        outs['scale_1'] = d


        return inter_feature,d
