import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self,channels):
        super(GroupNorm,self).__init__()
        self.gn = nn.GroupNorm(num_groups=32 , num_channels=channels , eps=1e-1 , affine=True)
        
    def forward(self,x):
        return self.gn(x)
    
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class ResidualBlock(nn.Module):
    def __init__(self,inPut,outPut):
        super(ResidualBlock,self).__init__()
        self.inPut = inPut
        self.outPut = outPut
        self.block = nn.Sequential(
            nn.Conv2d(inPut,outPut, 3, 1, 1),
            GroupNorm(outPut),
            Swish(),
            nn.Conv2d(outPut,outPut, 3, 1, 0)
        )
        
        if inPut!=outPut:
            self.channel_up = nn.Conv2d(inPut,outPut,1,1,0)
            
    def forward(self,x):
        if self.inPut!=self.outPut:
            return self.channel_up(x) + self.block(x)
        else :
            return x + self.block(x)
        
class UpSampletBlock(nn.Module):
    def __init__(self,channels):
        super(UpSampletBlock,self).__init__()
        self.conv = nn.Conv2d(channels,channels,3,1,1)
        
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2.0)
        return self.conv(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self,channels):
        super(DownSampleBlock,self).__init__()
        self.conv = nn.Conv2d(channels,channels,3,2,0)
        
    def forward(self,x):
        pad=(0,1,0,1)
        x=F.pad(x,pad,mode="constant",value=0)
        return self.conv(x)
    
#     attention part
class NonLocalBlock(nn.Module):
    def __init__(self,channels):
        super(NonLocalBlock,self).__init__()
        self.channels = channels
        
        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels,channels,1,1,0)
        self.v = nn.Conv2d(channels,channels,1,1,0)
        self.k = nn.Conv2d(channels,channels,1,1,0)
        self.proj_out = nn.Conv2d(channels,channels,1,1,0)
        
    def forward(self,x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b,c,h,w = q.shape
        
        q = q.reshape(b,c,h*w)
        q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        v = v.reshape(b,c,h*w)
        
        attn = torch.bmm(q,k)
        attn = attn*(int(c)**(-0.5))
        attn = F.softmax(attn,dim=2)
        attn = attn.permute(0,2,1)
        
        A = torch.bmm(v,attn)
        A = A.reshape(b,c,h,w)
        
        return x+A

class Encoder(nn.Module):
    def __init_(self,args):
        super(Encoder,self).__init__()
        channels = [128,128,128,256,256,256]
        attn_resolution = [16]
        num_ress_blocks = 2
        resolution = 256 #分辨率
        layers = [nn.Conv2d(args.image_channels[0],3,1,1)]
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            for j in range(num_ress_blocks):
                layers.append(ResidualBlock(in_channels,out_channels)
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i!=len(channels)-2:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //=2
            
        layers.append(ResidualBlock(channels[-1],channels[-1])
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1],channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1],args.latent_dim,3,1,1))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder,self).__init__()
        channels = [512,256,256,128,128]
        attn_resolution = [16]
        num_res_blocks = 3
        resolution = 16
        
        in_channels = channels[0]
        layers = [nn.Conv2d(args.latent_dim,in_channels,3,1,1),
                 ResidualBlock(in_channels,in_channels),
                 NonLocalBlock(in_channels),
                  ResidualBlock(in_channels,in_channels)
                 ]
        
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels,out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            if i!=0:
                layers.append(UpSampletBlock(in_channels))
                resolution = 2*resolution
            
        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels,args.image_channels,3,1,1))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.model(x)
    
class Codebook(nn.Module):
    def __init__(self,args):
        super(Codebook,self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        
        self.embedding = nn.Embedding(self.num_codebook_vectors,self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0/self.num_codebook_vectors,1.0/self.num_codebook_vectors)
        
        
    def forward(self,z):
        z = z.permute(0,2,3,1).contiguous()
        z_flattened = z.view(-1,self.latent_dim)
        
        d = torch.sum(z_flattened**2 ,dim=1,keepdim=True)+\
            torch.sum(self.embedding.weight**2,dim=1)-\
            2*(torch.matmul(z_flattened,self.embedding.weight.t()))
        
        min_encoding_indices = torch.argmin(d,dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_q.detach()-2)**2) + self.beta*torch.mean((z_q-z.detach())**2)
        
        z_q = z+(z_q-z).detach()
        z_q = z_q.permute(0,3,1,2)
        
        return z_q, min_encoding_indices,loss
