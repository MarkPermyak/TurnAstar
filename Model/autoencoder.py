import torch
from torch import nn
import pytorch_lightning as pl

from .modules.encoder import Encoder
from .modules.decoder import Decoder
from .modules.attention import SpatialTransformer
from .modules.pos_emb import PosEmbeds

class Autoencoder(pl.LightningModule):
    def __init__(self, 
                in_channels=2, 
                out_channels=1, 
                hidden_channels=64,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(64, 64),
                mode='f',
                *args,
                **kwargs):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.transformer = SpatialTransformer(
            hidden_channels, 
            attn_heads,
            heads_dim,
            attn_blocks, 
            attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)
        
        self.recon_criterion = nn.L1Loss() if mode == 'h' else nn.MSELoss()
        self.mode = mode
        self.k = 64*64 if mode == 'h' else 1
        
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x

    def step(self, batch, batch_idx, regime):
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self(inputs)

        loss = self.recon_criterion((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f'{regime}_recon_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        
        loss = self.step(batch, batch_idx, 'train')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
                 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0004)
        
        return [optimizer]