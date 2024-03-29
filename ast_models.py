# ######################## Installations ###############################
import torch
import os, wget
import torch, timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.spatial.distance import mahalanobis
from timm.models.layers import to_2tuple, trunc_normal_
# #####################################################################

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        return x

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, y):
        
        diff = input1 - input2
        
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / input1.size()[0]
        
        return loss

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=35, fstride=10, tstride=10, input_fdim=128, input_tdim=512, 
                 imagenet_pretrain=False, audioset_pretrain=False, model_path='/home/almogk/FSL_TL_E_C/pretraind_models/audioset_10_10_0.4593.pth', verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, 
                                                                            self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, 
                                                                                                                                 self.oringal_hw, self.oringal_hw)
                
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
                print('------------------### imagenet pretrained positional embedding is used###------------------.')
            else:
                
                # initialize all weight parameters
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1)
                        
                # if not use imagenet pretrained model, just randomly initialize a learnable embeddings
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, mean=0, std=0.5)
                if verbose == True:
                    print('------------------### randomly initialize positional embedding is used###------------------.')
                
        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(model_path) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='/home/almogk/FSL_TL_E_C/pretraind_models/audioset_10_10_0.4593.pth')
            
            sd = torch.load(model_path, map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, 
                                   audioset_pretrain=False, model_path=model_path, verbose=False)
            
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
           
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            print('------------------### imagnet + audioset pretrained positional embedding is used###------------------.')

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
       
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
       
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        vec_emmbeding_norm = x.detach().clone()
        
        x = self.mlp_head(x)
        
        return x, vec_emmbeding_norm

class Siamese_ASTModel(nn.Module):
    """
    SiameseASTModel extends the ASTModel class to implement a Siamese architecture
    for embedding pairs of spectrograms.
    """
    def __init__(self, input_tdim, con_los,  fc, fin, imagenet_pretrain, audioset_pretrain, checkpoint_path=None):
        super(Siamese_ASTModel, self).__init__()
        
        self.fc = fc
        self.con_los = con_los
        if checkpoint_path != None:
            ast_mdl = ASTModel(label_dim=35, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
            print(f'[*INFO] load {fin} checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            
            ast_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
            ast_model.load_state_dict(checkpoint)
            ast_model = ast_model.to(torch.device("cuda:0"))
            self.v = ast_model.module.v
            
        else:
            ast_mdl = ASTModel(label_dim=35, input_tdim=input_tdim, imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain)
            print(f'[*INFO] load {fin}  no checkpoint')
            ast_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
            ast_model = ast_model.to(torch.device("cuda:0"))
            self.v = ast_model.module.v
            
            
        if self.con_los:
           pass
        else:    
            if self.fc == 's':
                # self.fc_s = nn.Linear(1536, 1)
                self.fc_s = nn.Sequential(nn.LayerNorm(1536), nn.Linear(1536, 1))
                
            if self.fc == 'm':
                # self.fc_m = nn.Linear(768, 1)
                self.fc_m = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 1))
    
    def forward_once(self, x):
        
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        
        for blk in self.v.blocks:
            x = blk(x)
            
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2        
        
        return x
            
    @autocast()
    def forward(self, x1, x2):
        """
        :param x1: the first input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :param x2: the second input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: a tuple containing the embeddings of the two inputs
        """
        x1_embedding = self.forward_once(x1)
        x2_embedding = self.forward_once(x2)
    
        
        if self.con_los:
            
            return x1_embedding, x2_embedding
        
        else:
            if self.fc == 's':
                output = torch.cat((x1_embedding, x2_embedding), 1)
                output = self.fc_s(output)
            
            if self.fc == 'm':
                
                # Multiply (element-wise) the feature vectors of the two images together, 
                # to generate a combined feature vector representing the similarity between the two.
                combined_features = x1_embedding * x2_embedding
                
                # Pass the combined feature vector through classification head to get similarity value in the range of 0 to 1.
                output = self.fc_m(combined_features)
            
            if self.fc == 'maha':
                
                # # Assuming x1_embedding and x2_embedding have shape (batch_size, embedding_dim)
                x1_embedding = x1_embedding.unsqueeze(2)  # Add an extra dimension (batch_size, embedding_dim, 1)
                x2_embedding = x2_embedding.unsqueeze(2)

                # Compute Mahalanobis distance
                diff = x1_embedding - x2_embedding
                
                # Calculate squared Mahalanobis distance
                mahalanobis_dist_sq = torch.matmul(diff.transpose(1, 2), diff).squeeze(2)
                
                # Apply min-max normalization to the distances
                normalized_dist = (mahalanobis_dist_sq - mahalanobis_dist_sq.min()) / (mahalanobis_dist_sq.max() - mahalanobis_dist_sq.min())

                # Invert the distances to get similarity scores
                output = 1 - normalized_dist
            
            return output