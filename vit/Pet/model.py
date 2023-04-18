import torch
import torch.nn as nn
from torch.nn.functional import softmax

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0.):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.lin2(x)
        x = self.dropout(x)
        return x
    

 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1./(dim**0.5)
        self.qkv_gen = torch.nn.ModuleList()
        self.dim = dim
 
        # One matrix to generate query, key and value simultaneuosly
        self.attention_dim = dim//num_heads
        for head in range(num_heads):
            self.qkv_gen.append(torch.nn.Linear(dim, 3*self.attention_dim))
        self.linear_transformation = torch.nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
 
    def forward(self, x):
        batch_size = x.shape[0]
        # batch x (H*W/(patch_size^2)+1) x embed_dim -> 
        # -> num_heads x batch x (H*W/(patch_size^2)+1) x attention_dim
        # print(self.qkv_gen[0](x))
        # print([self.qkv_gen[head_i](x) for head_i in range(self.num_heads)])
        qkv = torch.stack(
            [self.qkv_gen[head_i](x) for head_i in range(self.num_heads)])
        # Unpack qkv into q, k, v
        q, k, v =\
        qkv[:, :, :, :self.attention_dim],\
        qkv[:, :, :, self.attention_dim:2*self.attention_dim],\
        qkv[:, :, :, 2*self.attention_dim:]
 
        # Get attention score (MatMul and scale steps)
        # num_heads x batch x (H*W/(patch_size^2)+1) x attention_dim @
        # num_heads x batch x attention_dim x (H*W/(patch_size^2)+1) ->
        # -> num_heads x batch x (H*W/(patch_size^2)+1) x (H*W/(patch_size^2)+1)
        normalized_attention_score = (q@k.transpose(-2, -1))*self.scale
 
        # Get weights for attention (Softmax step)
        softmaxed_attention_score = softmax(normalized_attention_score, dim = -1)
 
        # num_heads x batch x (H*W/(patch_size^2)+1) x attention_dim
        result = softmaxed_attention_score @ v
 
        # num_heads x batch x (H*W/(patch_size^2)+1) x attention_dim ->
        # batch x (H*W/(patch_size^2)+1) x num_heads x attention_dim
        result = torch.permute(result, dims = (1, 2, 0, 3))
 
        # batch x (H*W/(patch_size^2)+1) x num_heads x attention_dim ->
        # batch x (H*W/(patch_size^2)+1) x embed_dim
        result = torch.reshape(result, (batch_size, -1, self.dim))
        x = self.linear_transformation(result)
        x = self.attn_dropout(x)
        return x
    
class ImgPatches(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, img):
        patches = self.patch_embed(img)
        patches = torch.flatten(patches, 2)
        patches = torch.transpose(patches, 2, 1)
        return patches
    
class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.norm_pre = nn.LayerNorm(dim)
        self.attention = Attention(dim, attn_dropout = drop_rate, 
                                   num_heads = num_heads)
        self.mlp = MLP(dim, mlp_ratio*dim, dim, dropout = drop_rate)
        self.norm_pos = nn.LayerNorm(dim)
        

    def forward(self, x):
        xq = self.norm_pre(x)
        xq = self.attention(xq)
        x = xq + x
        xn = self.norm_pos(x)
        xn = self.mlp(xn)
        return x + xn
    
class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 drop_rate=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_patches = ImgPatches(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.cl = nn.Parameter(torch.ones((1, 1, embed_dim)))
        self.pos = nn.Parameter(torch.ones((1, 197, embed_dim)))
        self.transform = Transformer(depth, embed_dim, num_heads, 
                                     mlp_ratio, drop_rate)
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.img_patches(x)
        x = torch.concat((x, self.cl.repeat(x.shape[0], 1, 1)), dim=1)
        x = x + self.pos.data
        x = self.transform(x)
        class_embeding = x[:, -1, :]
        class_embeding = self.classifier(class_embeding)
        
        return class_embeding
    
a = torch.ones(1,3,224,224).cuda()
v = ViT().cuda()

b = v(a)

print(b.shape)