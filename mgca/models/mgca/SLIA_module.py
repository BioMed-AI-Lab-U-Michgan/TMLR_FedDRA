import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist
from transformers import ViTModel, ViTConfig
from transformers import BertModel, BertConfig
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)

    
    
    
class FedDRA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        ########################################################################
        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.global_img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        self.global_text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        

        config = BertConfig.from_pretrained('bert-base-uncased')
        self.align_text_model = BertModel(config).encoder.layer[-1]
        self.global_align_text_model = BertModel(config).encoder.layer[-1]
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        self.align_img_model = ViTModel(config).encoder.layer[-1]
        self.global_align_img_model = ViTModel(config).encoder.layer[-1]
        
        
        ########################################################################
        self.adapt = 0
        self.client_ratio = 1
        self.client_index = 0
        self.client_ratio_list = torch.ones(self.hparams.n_client)/self.hparams.n_client

        self.last_log = None

        ########################################################################
    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        if self.adapt == 1:
            img_feat_q_init, patch_feat_q_init = self.img_encoder_q(
                batch["imgs"])
            
            report_feat_q_init, word_feat_q_init, word_attn_q, sents = self.text_encoder_q(
                batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
            with torch.no_grad():
                global_img_feat_q_init, global_patch_feat_q_init = self.global_img_encoder_q(
                    batch["imgs"])
                global_report_feat_q_init, global_word_feat_q_init, global_word_attn_q, global_sents = self.global_text_encoder_q(
                    batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        else:
            with torch.no_grad():
                img_feat_q_init, patch_feat_q_init = self.img_encoder_q(
                    batch["imgs"])
                global_img_feat_q_init, global_patch_feat_q_init = self.global_img_encoder_q(
                    batch["imgs"])
                report_feat_q_init, word_feat_q_init, word_attn_q, sents = self.text_encoder_q(
                    batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
                global_report_feat_q_init, global_word_feat_q_init, global_word_attn_q, global_sents = self.global_text_encoder_q(
                    batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        
        
        
        all_feat_q = self.align_img_model(torch.concat([img_feat_q_init.unsqueeze(1), patch_feat_q_init], dim = 1))[0]
        img_feat_q = all_feat_q[:,0].contiguous()
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        all_feat_q = self.align_text_model(torch.concat([report_feat_q_init.unsqueeze(1), word_feat_q_init], dim = 1))[0]
        report_feat_q = all_feat_q[:,0].contiguous()
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        
        
        with torch.no_grad():
            global_all_feat_q = self.global_align_img_model(torch.concat([global_img_feat_q_init.unsqueeze(1), global_patch_feat_q_init], dim = 1))[0]
            global_img_feat_q = global_all_feat_q[:,0].contiguous()
            global_img_emb_q = self.global_img_encoder_q.global_embed(global_img_feat_q)
            global_all_feat_q = self.global_align_text_model(torch.concat([global_report_feat_q_init.unsqueeze(1), global_word_feat_q_init], dim = 1))[0]
            global_report_feat_q = global_all_feat_q[:,0].contiguous()
            global_report_emb_q = self.global_text_encoder_q.global_embed(global_report_feat_q)


        

        img_emb_q = F.normalize(img_emb_q, dim=-1)
        global_img_emb_q = F.normalize(global_img_emb_q, dim=-1)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        global_report_emb_q = F.normalize(global_report_emb_q, dim=-1)
        
        diag_cos = F.cosine_similarity(report_emb_q, img_emb_q)
        g_diag_cos = F.cosine_similarity(global_report_emb_q, global_img_emb_q)


        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()
        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.softmax_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_ita = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
            
        loss_align = F.mse_loss(global_img_emb_q, img_emb_q)
        loss_align = loss_align + F.mse_loss(global_report_emb_q, report_emb_q)


            
        return loss_ita, loss_align, acc1, acc5, g_diag_cos.mean(), diag_cos.mean()
    
    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.hparams.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.hparams.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def training_step(self, batch, batch_idx):
        
        
        loss_ita, loss_align, acc1, acc5, g_diag_cos, diag_cos = self(
            batch, batch_idx, "train")
        loss = (self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * loss_align) * self.client_ratio
        
        current_step = self.global_step
        log = {
            "train_loss": loss,
            "train_loss_align": loss_align,
            "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "g_diag_cos_mean": g_diag_cos,
            "diag_cos_mean": diag_cos,
            "current_step:": current_step
        }
        self.last_log = log
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            pass

    def validation_step(self, batch, batch_idx):
        loss_ita, loss_align, acc1, acc5, g_diag_cos, diag_cos = self(
            batch, batch_idx, "valid")
        
        loss = self.hparams.lambda_1 * loss_ita + loss_align
        
        log = {
            "val_loss": loss,
            "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_align": loss_align,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "g_diag_cos_mean": g_diag_cos,
            "diag_cos_mean": diag_cos
        }
        
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        loss_ita, loss_align, acc1, acc5, g_diag_cos, diag_cos = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_1 * loss_ita + loss_align

        log = {
            "test_loss": loss,
            "test_loss_ita": self.hparams.lambda_1 * loss_ita,
            "test_loss_align": loss_align,
            "test_acc1": acc1,
            "test_acc5": acc5,
            "g_diag_cos_mean": g_diag_cos,
            "diag_cos_mean": diag_cos
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss
    
    def test_epoch_end(self, test_step_outputs):

        avg_loss = torch.stack([x for x in test_step_outputs]).mean()
        self.test_loss = avg_loss
        return {"loss":avg_loss}
    
    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        
       
        return optimizer
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72)
        parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--commu_current", type=int, default=0)
        parser.add_argument("--wandb_id", type=str, default="ICML_BEST_PAPER")
        return parser

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        print("dataset_size:",dataset_size,"effective_batch_size:",effective_batch_size)
        return (dataset_size // effective_batch_size) 
    