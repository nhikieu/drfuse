import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from .layers import general_conv2d_prenorm, fusion_prenorm_2d
from torch import optim
from .criterions_rs import *
import wandb
import os
import numpy as np
from tqdm import tqdm
import time
import json

BASIC_DIMS = 8
TRANSFORMER_BASIC_DIMS = 512
MLP_DIM = 4096
NUM_HEADS = 8
DEPTH = 1
NUM_MODALS = 2
BETA1 = 0.5
LEARNING_RATE=2e-4
patch_size = 32 # 512 / 16, change when input img size changes

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / (1e-6)

class DrFuse_RS_Trainer(nn.Module):
    def __init__(self, config, pretrain=False, ckpt=None):
        super().__init__()
        self.pretrain = pretrain
        self.model = DrFuse_RS_Model()
        self.alignment_cos_sim = nn.CosineSimilarity(dim=1)
        self.jsd = JSD()
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=6e-4, weight_decay=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2)

        if ckpt is not None:
            ckpt = torch.load(ckpt)
            self.model.load_state_dict(ckpt['model_state_dict'])

        self.best_loss = math.inf
        self.best_model_path = ''
        self.saveModelPath = config['checkpoint_pth']
        os.makedirs(self.saveModelPath)

        self.example_count = 0
        self.loss_dict = {
            'train': {
                'pred_final': [],
                'pred_rgb': [],
                'pred_ndsm': [],
                'pred_shared': [],
                'scale_loss': [],
                'sim_rgb': [],
                'sim_ndsm': [],
                'jsd': []
            },
            'val': {
                'pred_final': [],
                'pred_rgb': [],
                'pred_ndsm': [],
                'pred_shared': [],
                'scale_loss': [],
                'sim_rgb': [],
                'sim_ndsm': [],
                'jsd': []
            }
        }

    def forward(self, train_loader, val_loader, device='cuda'):
        wandb.watch(self.model, log="all", log_freq=100)
        self.model.to(device)

        for epoch in tqdm(range(150)):
            for _, (images, labels, masks) in enumerate(train_loader):
                batch = (images.to(device), labels.to(device), torch.tensor(masks).to(device))

                if self.example_count % 512 == 0 or (_+1) == len(train_loader):
                    log = True
                else:
                    log = False

                train_loss = self.training_step(batch, log, self.pretrain)
                self.example_count += len(images)

                if self.example_count % 128 == 0 or (_+1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                # Report metrics
                if self.example_count % 512 == 0 or (_+1) == len(train_loader):
                    running_val_loss = 0
                    with torch.no_grad():
                        for _, (images, labels, masks) in enumerate(val_loader):
                            batch = (images.to(device), labels.to(device), torch.tensor(masks).to(device))
                            val_loss = self.validation_step(batch, self.pretrain)
                            running_val_loss += val_loss

                            if (_+1) == len(val_loader):
                                viz = self._log_val_seg(batch, self.pretrain)

                    val_loss = running_val_loss / len(val_loader)

                    self._wanb_log(train_loss, val_loss, self.example_count, epoch, viz)

                    # Save loss_dict to file
                    with open(os.path.join(self.saveModelPath,'loss_dict.json'), 'w') as f:
                        json.dump(self.loss_dict, f)

                    if val_loss <= self.best_loss:
                        self.best_loss = val_loss

                        # if there are more than 5 saved checkpoints
                        dir = os.listdir(self.saveModelPath)
                        full_path = [os.path.join(self.saveModelPath, x) for x in dir]
                        if len(dir) >= 5:
                            # remove oldest checkpoint b4 saving a new one
                            oldest_file = min(full_path, key=os.path.getctime)
                            os.remove(oldest_file)
                        
                        # Save best model
                        f_name = '_'.join([str(int(time.time())), f'loss{self.best_loss:.4f}'])
                        best_model_path = os.path.join(self.saveModelPath, f_name)
                        torch.save(
                            {
                                'model_state_dict': self.model.state_dict()
                            },
                            best_model_path
                        )

    def training_step(self, batch, log, pretrain):
        (x, y, masks) = batch
        out = self.model(x, masks)

        train_loss = self._compute_and_log_loss(out, y_gt=y, log=log, pretrain=pretrain)

        train_loss.backward()
        
        return train_loss.item()
    
    def validation_step(self, batch, pretrain):
        (x, y, masks) = batch
        out = self.model(x, masks)
        val_loss = self._compute_and_log_loss(out, y_gt=y, log=True, mode='val', pretrain=pretrain)

        return val_loss.item()
    
    def _wanb_log(self, train_loss, val_loss, step, epoch, viz):
        if len(viz) == 3:
            val_img, val_seg_full, val_seg_miss = viz
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_img': val_img,
                'val_seg_full': val_seg_full,
                'val_seg_miss': val_seg_miss
            }, step=step)
        else:
            val_img, val_seg_rgb, val_seg_ndsm, val_seg_shared = viz
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_img': val_img,
                'val_seg_rgb': val_seg_rgb,
                'val_seg_ndsm': val_seg_ndsm,
                'val_seg_shared': val_seg_shared
            }, step=step)

        print(f"Train loss after {str(step).zfill(5)} examples: {train_loss:.3f}")
        print(f"Val loss after {str(step).zfill(5)} examples: {val_loss:.3f}")

    def _log_val_seg(self, batch, pretrain):
        (x, y, masks) = batch
        x = x.to('cpu')
        self.model.to('cpu')

        img = x[0].unsqueeze(0)
        rgb_img = x[0].unsqueeze(0)[:, 0:3, :, :]
        img_array = rgb_img.permute((0, 2, 3, 1)).numpy().squeeze()

        if pretrain:
            with torch.no_grad():
                mask = torch.tensor([1])
                output = self.model(img, mask)
                output_rgb = output['pred_rgb'][0]
                output_ndsm = output['pred_ndsm'][0]
                output_shared = output['pred_shared'][0]

            output_rgb = output_rgb.argmax(dim=0)
            output_ndsm = output_ndsm.argmax(dim=0)
            output_shared = output_shared.argmax(dim=0)

            seg_array_rgb = self._convert_prediction(output_rgb.squeeze())
            seg_array_ndsm = self._convert_prediction(output_ndsm.squeeze())
            seg_array_shared = self._convert_prediction(output_shared.squeeze())
        
            img_array = wandb.Image(img_array, caption="images")
            seg_array_rgb = wandb.Image(seg_array_rgb, caption="seg_rgb")
            seg_array_ndsm = wandb.Image(seg_array_ndsm, caption="seg_ndsm")
            seg_array_shared = wandb.Image(seg_array_shared, caption="seg_shared")
            
            self.model.to('cuda')
            return (img_array, seg_array_rgb, seg_array_ndsm, seg_array_shared)
        else:
            with torch.no_grad():
                mask = torch.tensor([1])
                output_full = self.model(img, mask)['pred_multimodal'][0]
                mask = torch.tensor([0])
                output_miss = self.model(img, mask)['pred_multimodal'][0]

            output_full = output_full.argmax(dim=0)
            output_miss = output_miss.argmax(dim=0)

            seg_array_full = self._convert_prediction(output_full.squeeze())
            seg_array_miss = self._convert_prediction(output_miss.squeeze())
        
            img_array = wandb.Image(img_array, caption="images")
            seg_array_full = wandb.Image(seg_array_full, caption="seg_full")
            seg_array_miss = wandb.Image(seg_array_miss, caption="seg_miss")

            self.model.to('cuda')

            return (img_array, seg_array_full, seg_array_miss)


    def _convert_prediction(self, image):
        # TODO make clase 6 - bg same color with class 0
        valGT = [[255,255,255], [0,0,255], [0,255,255], [0,255,0], [255,255,0], [255,255,255]]

        output = np.zeros((image.shape[0], image.shape[1], 3))
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                class_idx = image[i, j]
                output[i,j,:] = valGT[class_idx]
            
        return output.astype('uint8')
    
    def _compute_prediction_losses(self, model_output, y_gt, pretrain, log=True, mode='train'):
        num_cls = 6
        y_gt = torch.squeeze(y_gt, dim=1)
        y_gt = expand_target(y_gt)
        y_gt = y_gt.type(torch.LongTensor).to('cuda' if torch.cuda.is_available() else 'mps')

        # TODO implement missing modality at training
        loss_pred_rgb = softmax_weighted_loss(model_output['pred_rgb'], y_gt, num_cls=num_cls)
        # loss_pred_rgb_ = dice_loss(model_output['pred_rgb'], y_gt, num_cls=num_cls)
        # loss_pred_rgb = loss_pred_rgb + loss_pred_rgb_

        loss_pred_ndsm = softmax_weighted_loss(model_output['pred_ndsm'], y_gt, num_cls=num_cls)
        # loss_pred_ndsm_ = dice_loss(model_output['pred_ndsm'], y_gt, num_cls=num_cls)
        # loss_pred_ndsm = loss_pred_ndsm + loss_pred_ndsm_

        loss_pred_shared = softmax_weighted_loss(model_output['pred_shared'], y_gt, num_cls=num_cls)
            # loss_pred_shared_ = dice_loss(model_output['pred_shared'], y_gt, num_cls=num_cls)
            # loss_pred_shared = loss_pred_shared + loss_pred_shared_

        if not pretrain:
            loss_pred_final = softmax_weighted_loss(model_output['pred_multimodal'], y_gt, num_cls=num_cls)
            # loss_pred_final_ = dice_loss(model_output['pred_multimodal'], y_gt, num_cls=num_cls)
            # loss_pred_final = loss_pred_final + loss_pred_final_

            scale_cross_loss = torch.zeros(1).float().to('cuda' if torch.cuda.is_available() else 'mps')
            scale_dice_loss = torch.zeros(1).float().to('cuda' if torch.cuda.is_available() else 'mps')
            for scale_pred in model_output['aux_preds']:
                scale_pred = scale_pred.to('cuda' if torch.cuda.is_available() else 'mps')
                y_gt = y_gt.to('cuda' if torch.cuda.is_available() else 'mps')
                scale_cross_loss += softmax_weighted_loss(scale_pred, y_gt, num_cls=num_cls)
                # scale_dice_loss += dice_loss(scale_pred, y_gt, num_cls=num_cls)
            scale_loss = scale_cross_loss + scale_dice_loss

            return loss_pred_rgb, loss_pred_ndsm, loss_pred_shared, loss_pred_final, scale_loss
        else:
            return loss_pred_rgb, loss_pred_ndsm, loss_pred_shared
    
    def _masked_abs_cos_sim(self, x, y):
        return (self.alignment_cos_sim(x, y).abs()).sum() / (1e-6)
    
    def _disentangle_loss_jsd(self, model_output, pairs, log=True, mode='train'):
        batch_size = model_output['feat_rgb_shared'].shape[0]
        loss_sim_rgb = self._masked_abs_cos_sim(model_output['feat_rgb_shared'].view(batch_size, -1),
                                                model_output['feat_rgb_distinct'].view(batch_size, -1))
        loss_sim_ndsm = self._masked_abs_cos_sim(model_output['feat_ndsm_shared'].view(batch_size, -1),
                                                model_output['feat_ndsm_distinct'].view(batch_size, -1))

        jsd = self.jsd(model_output['feat_rgb_shared'].sigmoid(),
                       model_output['feat_ndsm_shared'].sigmoid())
        
        loss_disentanglement = jsd + loss_sim_rgb + loss_sim_ndsm

        if log:
            self.loss_dict[mode]['sim_rgb'].append(loss_sim_rgb.item())
            self.loss_dict[mode]['sim_ndsm'].append(loss_sim_ndsm.item())
            self.loss_dict[mode]['jsd'].append(jsd.item())

        return loss_disentanglement
    
    def _compute_and_log_loss(self, model_output, y_gt, pretrain, log=False, mode='train', pairs=None):
        # TODO handle missing modality with pairs, search for where 'pairs' argument is passed in original drfuse

        prediction_losses = self._compute_prediction_losses(model_output, y_gt, pretrain, log, mode)

        if log:
            self.loss_dict[mode]['pred_rgb'].append(prediction_losses[0].item())
            self.loss_dict[mode]['pred_ndsm'].append(prediction_losses[1].item())
            self.loss_dict[mode]['pred_shared'].append(prediction_losses[2].item())
            if not pretrain:
                self.loss_dict[mode]['pred_final'].append(prediction_losses[3].item())
                self.loss_dict[mode]['scale_loss'].append(prediction_losses[4].item())
        
        loss_prediction = sum(prediction_losses)

        # if not pretrain:
        loss_disentanglement = self._disentangle_loss_jsd(model_output, log, mode)

        loss_total = loss_prediction + loss_disentanglement

            # TODO aux loss for attention ranking

        return loss_total
        # else:
        #     return loss_prediction
    

class DrFuse_RS_Model(nn.Module):
    def __init__(self, num_cls=6):
        super().__init__()
        self.rgb_encoder = Encoder(in_channels=3)
        self.ndsm_encoder = Encoder()

        self.rgb_encode_conv = nn.Conv2d(BASIC_DIMS*16, TRANSFORMER_BASIC_DIMS, kernel_size=1, stride=1, padding=0)
        self.ndsm_encode_conv = nn.Conv2d(BASIC_DIMS*16, TRANSFORMER_BASIC_DIMS, kernel_size=1, stride=1, padding=0)

        self.rgb_pos = nn.Parameter(torch.zeros(1, patch_size**2, TRANSFORMER_BASIC_DIMS))
        self.ndsm_pos = nn.Parameter(torch.zeros(1, patch_size**2, TRANSFORMER_BASIC_DIMS))

        self.rgb_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.ndsm_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)

        self.rgb_shared_feat = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.ndsm_shared_feat = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)

        self.rgb_distinct_feat = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.ndsm_distinct_feat = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)

        self.rgb_decode_conv = nn.Conv2d(TRANSFORMER_BASIC_DIMS, BASIC_DIMS*16, kernel_size=1, stride=1, padding=0)
        self.ndsm_decode_conv = nn.Conv2d(TRANSFORMER_BASIC_DIMS, BASIC_DIMS*16, kernel_size=1, stride=1, padding=0)
        self.shared_decode_conv = nn.Conv2d(TRANSFORMER_BASIC_DIMS, BASIC_DIMS*16, kernel_size=1, stride=1, padding=0)

        self.rgb_decoder_sep = Decoder_sep(num_cls=num_cls)
        self.ndsm_decoder_sep = Decoder_sep(num_cls=num_cls)
        self.shared_decoder_sep = DecoderFuse(num_cls=num_cls, num_modals=1)

        self.shared_project = nn.Sequential(
            nn.Linear(TRANSFORMER_BASIC_DIMS, TRANSFORMER_BASIC_DIMS*2),
            nn.ReLU(),
            nn.Linear(TRANSFORMER_BASIC_DIMS*2, TRANSFORMER_BASIC_DIMS),
            nn.ReLU(),
            nn.Linear(TRANSFORMER_BASIC_DIMS, TRANSFORMER_BASIC_DIMS)
        )

        self.multimodal_transformer = Transformer(embedding_dim=TRANSFORMER_BASIC_DIMS, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM)
        self.num_modals = 3
        self.multimodal_decode_conv = nn.Conv2d(TRANSFORMER_BASIC_DIMS*self.num_modals, BASIC_DIMS*16*self.num_modals, kernel_size=1, padding=0)
        self.fuse_decoder = DecoderFuse(num_cls=num_cls)

        self._seg_layer_features = None


    def forward(self, x, masks):
        batch_size = x.shape[0]

        # TODO implement missing modality at training
        rgb_inputs = x[:, 0:3, :, :]
        ndsm_inputs = x[:, 3:4, :, :]

        rgb_x1, rgb_x2, rgb_x3, rgb_x4, rgb_x5 = self.rgb_encoder(rgb_inputs)
        ndsm_x1, ndsm_x2, ndsm_x3, ndsm_x4, ndsm_x5 = self.ndsm_encoder(ndsm_inputs)

        rgb_token_x5 = self.rgb_encode_conv(rgb_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)
        ndsm_token_x5 = self.ndsm_encode_conv(ndsm_x5).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, TRANSFORMER_BASIC_DIMS)

        rgb_transformer_x5 = self.rgb_transformer(rgb_token_x5, self.rgb_pos)
        ndsm_transformer_x5 = self.ndsm_transformer(ndsm_token_x5, self.ndsm_pos)

        rgb_shared_feat = self.rgb_shared_feat(rgb_transformer_x5)
        rgb_distinct_feat = self.rgb_distinct_feat(rgb_transformer_x5)

        ndsm_shared_feat = self.ndsm_shared_feat(ndsm_transformer_x5)
        ndsm_distinct_feat = self.ndsm_distinct_feat(ndsm_transformer_x5)

        # Modality-specific heads
        rgb_token_x5 = self.rgb_decode_conv(rgb_distinct_feat.permute(0, 2, 1).contiguous().view(batch_size, TRANSFORMER_BASIC_DIMS, patch_size, patch_size))
        ndsm_token_x5 = self.ndsm_decode_conv(ndsm_distinct_feat.permute(0, 2, 1).contiguous().view(batch_size, TRANSFORMER_BASIC_DIMS, patch_size, patch_size))
        rgb_pred = self.rgb_decoder_sep(rgb_x1, rgb_x2, rgb_x3, rgb_x4, rgb_token_x5)
        ndsm_pred = self.ndsm_decoder_sep(ndsm_x1, ndsm_x2, ndsm_x3, ndsm_x4, ndsm_token_x5)

        rgb_shared_feat = self.shared_project(rgb_shared_feat)
        ndsm_shared_feat = self.shared_project(ndsm_shared_feat)

        # 0 if only ndsm is available, 1 if both modalities are available
        pairs = masks.unsqueeze(1).unsqueeze(1)

        h1 = rgb_shared_feat # (6, 1024, 512)
        h2 = ndsm_shared_feat
        term1 = torch.stack([h1+h2, h1+h2, h1, h2], dim=2)
        term2 = torch.stack([torch.zeros_like(h1), torch.zeros_like(h1), h1, h2], dim=2)
        feat_avg_shared = torch.logsumexp(term1, dim=2) - torch.logsumexp(term2, dim=2)

        feat_avg_shared = pairs * feat_avg_shared + (1 - pairs) * ndsm_shared_feat

        feat_avg_shared_ = feat_avg_shared.view(batch_size, patch_size, patch_size, TRANSFORMER_BASIC_DIMS).permute(0, 3, 1, 2).contiguous()
        feat_avg_shared_ = self.shared_decode_conv(feat_avg_shared_)
        shared_pred, _ = self.shared_decoder_sep(feat_avg_shared_)

        # TODO Disease-wise Attention
        rgb_distinct_feat = pairs * rgb_distinct_feat # assume ndsm is always available
        multimodal_token_x5 = torch.cat((rgb_distinct_feat, ndsm_distinct_feat, feat_avg_shared), dim=1)
        multimodal_token_x5 = self.multimodal_transformer(multimodal_token_x5)
        multimodal_token_x5 = self.multimodal_decode_conv(multimodal_token_x5.view(batch_size, patch_size, patch_size, TRANSFORMER_BASIC_DIMS*self.num_modals).permute(0, 3, 1, 2).contiguous())
        multimodal_pred, aux_preds = self.fuse_decoder(multimodal_token_x5)
        
        return {'pred_rgb': rgb_pred, 'pred_ndsm': ndsm_pred, 'pred_shared': shared_pred, 'pred_multimodal': multimodal_pred, 'aux_preds': aux_preds,
                'feat_rgb_shared': rgb_shared_feat, 'feat_ndsm_shared': ndsm_shared_feat,
                'feat_rgb_distinct': rgb_distinct_feat, 'feat_ndsm_distinct': ndsm_distinct_feat}
    
    @property
    def seg_layer_features(self):
        return self.fuse_decoder.seg_layer_features.detach()
    

class DecoderFuse(nn.Module):
    def __init__(self, num_cls=6, num_modals=3):
        super().__init__()
        self.d4_c1 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv2d(in_channels=BASIC_DIMS*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv2d(in_channels=BASIC_DIMS*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv2d(in_channels=BASIC_DIMS*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(in_channels=BASIC_DIMS*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv2d(in_channels=BASIC_DIMS, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.RFM5 = fusion_prenorm_2d(in_channel=BASIC_DIMS*16, num_modals=num_modals)

    def forward(self, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.d4_out(self.d4_c2(de_x5))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.d3_out(self.d3_c2(de_x4))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.d2_out(self.d2_c2(de_x3))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.d1_out(self.d1_c2(de_x2))
        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        self._seg_layer_features = de_x1.detach()

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))
    
    @property
    def seg_layer_features(self):
        return self._seg_layer_features.detach()


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv2d(in_channels=in_channels, out_channels=BASIC_DIMS, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, pad_type='reflect')
        self.e1_c3 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, pad_type='reflect')

        self.e2_c1 = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, pad_type='reflect')
        self.e2_c3 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, pad_type='reflect')

        self.e3_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, pad_type='reflect')
        self.e3_c3 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, pad_type='reflect')

        self.e4_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, pad_type='reflect')
        self.e4_c3 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, pad_type='reflect')

        self.e5_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*16, pad_type='reflect')
        self.e5_c3 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5
    

class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos=None):
        for j in range(self.depth):
            if pos is not None:
                x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d4_c1 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(BASIC_DIMS*16, BASIC_DIMS*8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3_c1 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(BASIC_DIMS*8, BASIC_DIMS*4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_c1 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(BASIC_DIMS*4, BASIC_DIMS*2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_c1 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(BASIC_DIMS*2, BASIC_DIMS, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(BASIC_DIMS, BASIC_DIMS, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv2d(in_channels=BASIC_DIMS, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

