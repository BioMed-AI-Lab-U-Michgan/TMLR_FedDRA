import datetime
import os
from argparse import ArgumentParser
import wandb
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
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist
from mgca.models.mgca.SLIA_module import FedDRA
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho=0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w
def project_onto_chi_square_ball(w, rho, tol = 1e-10):
    assert (rho > 0)
    rho = float(rho)
  
    # sort in decreasing order
    w_sort = np.sort(w) # increasing
    w_sort = w_sort[::-1] # decreasing
  
    w_sort_cumsum = w_sort.cumsum()
    w_sort_sqr_cumsum = np.square(w_sort).cumsum()
    nn = float(w_sort.shape[0])

    lam_min = 0.0
    lam_max = (1/nn) * (nn * w_sort[0] / np.sqrt(2. * rho + 1.) - 1.)
    lam_init_max = lam_max

    if (lam_max <= 0): # optimal lambda is 0
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, 0., rho)
        p = w - eta
        low_inds = p < 0
        p[low_inds] = 0.
        return p

    # bisect on lambda to find the optimal lambda value
    while (lam_max - lam_min > tol * lam_init_max):
        lam = .5 * (lam_max + lam_min)
        (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)

        # compute norm(p(lam))^2 * (1+lam * nn)^2
        thresh = .5 * nn * (w_sort_sqr_cumsum[ind] - 2. * eta * w_sort_cumsum[ind] + eta**2 * (ind+1.))
        if (thresh > (rho + .5) * (1 + lam * nn)**2):
            # constraint infeasible, increase lam (dual var)
            lam_min = lam
        else:
            # constraint loose, decrease lam (dual var)
            lam_max = lam

    lam = .5 * (lam_max + lam_min)
    (eta, ind) = solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho)
    p = w - eta
    low_inds = p < 0
    p[low_inds] = 0
    return (1. / (1. + lam * nn)) * p
def solve_inner_eta(w_sort, w_sort_cumsum, nn, lam, rho):
    fs = w_sort - (w_sort_cumsum - (1. + lam * nn)) / (np.arange(nn) + 1.)
    ind = (fs > 0).sum()-1
    return ((1 / (ind+1.)) * (w_sort_cumsum[ind] - (1. + lam * nn)), ind)
def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    print("This is main process!")
    return True

def avg_param(server_model, client_list):
    from collections import OrderedDict 
    worker_state_dict = [x.state_dict() for x in client_list]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(client_list)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(client_list)
    server_model.load_state_dict(fed_state_dict)
    return server_model

def cli_main(args, callbacks, ckpt_dir, model, datamodule, client_index = -1, is_adapt = 0, client_ratio = 1):
    model.adapt = is_adapt
    model.client_ratio = client_ratio
    print(f"Started client{client_index}'s training!")
    trainer = Trainer.from_argparse_args(
                    args=args,
                    callbacks=callbacks)
    model.training_steps = model.num_training_steps(trainer, datamodule)
    trainer.fit(model, datamodule=datamodule)
    print(f"finished client{client_index}'s training!")
    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)
    from copy import deepcopy
    val_result = model.last_log
    
    model.to("cpu")
    
    del trainer
    import gc
    gc.collect()
    
    torch.cuda.empty_cache()
    return model
def test_server(args, callbacks, model, datamodule):
    trainer = Trainer.from_argparse_args(
                    args=args,
                    callbacks=callbacks)
    print(f"validate on client{datamodule.client_index}")
    test_result = trainer.test(model, datamodule)
    model.to("cpu")
    
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return test_result
def FedPretrain():
    #################### setting args #############################
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = FedDRA.add_model_specific_args(parser)
    args = parser.parse_args()
    
    ###############################################################
    args.deterministic = True
    args.n_client = 5
    args.gamma = 0.01
    args.num_com = 25
    args.step_per_com = 50
    args.max_steps = args.step_per_com
    
    # seed_everything(args.seed)
    print(f"Current commu step is {args.commu_current}")
    
    ###############################################################
    store_current_device = -1
    if torch.cuda.is_available():
        store_current_device = torch.cuda.current_device()
        
    ####################### initialize server #####################
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/server")
    ckpt_path = os.path.join(
        BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/server/last.pth")
    copy_path = os.path.join(
        BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/server/last_copy.pth") 
    last_time_path = os.path.join(
        BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/server/epoch=0-step={args.step_per_com-1}.pth") 
    
    if os.path.exists(last_time_path):
        os.remove(last_time_path)
    if args.commu_current==0:
        if os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        
    
    if os.path.exists(ckpt_path):
        server = FedDRA(**args.__dict__)
        
        server.load_state_dict(torch.load(ckpt_path)) # just for model parameters
        ############### remake ###############
        
        os.rename(ckpt_path, copy_path)
        
        ######################################
    else:
        if os.path.exists(copy_path):
            server = FedDRA(**args.__dict__)
            server.load_state_dict(torch.load(copy_path))
            print("server copy path's ckpt is loaded!")
        else:
            server = FedDRA(**args.__dict__)

    
    ################################################################
    datamodule_list = []
    model_list = []

    callbacks_list = []
    ckpt_dir_list = []
    
    for i in range(args.n_client):
        client_index = i
        datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers, 224, client_index, "MultimodalPretrainingDataset")

        
        ckpt_dir = os.path.join(
            BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/client{i}")
        # Add load from checkpoint
        ckpt_path = os.path.join(
            BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/client{i}/last.ckpt")
        copy_path = os.path.join(
            BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/client{i}/last_copy.ckpt")
        last_time_path  = os.path.join(
            BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/client{i}/epoch=0-step={args.step_per_com-1}.ckpt")
        
        if os.path.exists(last_time_path):
            os.remove(last_time_path)
        if args.commu_current==0:
            if os.path.exists(ckpt_dir):
                import shutil
                shutil.rmtree(ckpt_dir)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
        
        if os.path.exists(ckpt_path):
            model = FedDRA.load_from_checkpoint(ckpt_path)
            print(f"client{i} has been loaded!")
            ############### remake ###############
            os.rename(ckpt_path, copy_path)
            
            ######################################
        else:
            if os.path.exists(copy_path):
                model = FedDRA.load_from_checkpoint(copy_path)
                print(f"client{i} copy path's ckpt is loaded!")
            else:
                model = FedDRA(**args.__dict__)
                print(f"client{i} has been initialized!")

        model.load_state_dict(server.state_dict()) # for the communication of FedAvg
        
        callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="train_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=0),
        ]
        
        #######################
        model_list.append(model)
        datamodule_list.append(datamodule)
        callbacks_list.append(callbacks)
        ckpt_dir_list.append(ckpt_dir)
         
    ########################## Federated training #########################
    
    ########################################################################
    server.to("cpu")
    
    for commu in range(args.commu_current, args.commu_current+1):
        #############################################################
        # 1st stage 
        ############################################################
        
        args.max_steps = 20

        for index, client in enumerate(model_list):
            new_model = cli_main(args, callbacks_list[index], ckpt_dir_list[index], client
                                      ,datamodule_list[index], index, is_adapt = 0, client_ratio = server.client_ratio_list[index])

            
            model_list[index] = new_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        ##############################################################
        # model aggregate
        #############################################################
        
        server = avg_param(server, model_list)
        server.global_align_img_model.load_state_dict(server.align_img_model.state_dict()) 
        server.global_align_text_model.load_state_dict(server.align_text_model.state_dict())
        server.global_img_encoder_q.global_embed.load_state_dict(server.img_encoder_q.global_embed.state_dict()) 
        server.global_text_encoder_q.global_embed.load_state_dict(server.text_encoder_q.global_embed.state_dict())
        
        
        
        #############################################################
        # 2nd stage 
        ############################################################

        args.max_steps = 50
        for index, client in enumerate(model_list):
            client.global_align_img_model.load_state_dict(server.global_align_img_model.state_dict()) 
            client.global_align_text_model.load_state_dict(server.global_align_text_model.state_dict())
            client.global_img_encoder_q.global_embed.load_state_dict(server.global_img_encoder_q.global_embed.state_dict()) 
            client.global_text_encoder_q.global_embed.load_state_dict(server.global_text_encoder_q.global_embed.state_dict())

            new_model = cli_main(args, callbacks_list[index], ckpt_dir_list[index], client
                                      ,datamodule_list[index], index, is_adapt = 1, client_ratio = server.client_ratio_list[index])

            model_list[index] = new_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            
        
        server = avg_param(server, model_list)
        server.global_img_encoder_q.load_state_dict(server.img_encoder_q.state_dict())
        server.global_text_encoder_q.load_state_dict(server.text_encoder_q.state_dict())

        print("Finished 2-stage training !!!!!!!!!!!!!!!!!!!!!!!")
        
        ##################### calculate weights #########################
        client_ratio_list = torch.ones(args.n_client)/args.n_client
        loss_list = []
        loss_sum = 0
        cos_mean = 0

        for index, client_data in enumerate(datamodule_list):
            trainer = Trainer.from_argparse_args(
                    args=args,
                    callbacks=callbacks_list[index])
            test_result = trainer.test(server, client_data)
            cos_mean = cos_mean + test_result[0]['diag_cos_mean']/len(datamodule_list)
            loss = server.test_loss
            loss = server.client_ratio_list[index] * torch.exp(args.step_per_com * loss * args.gamma)
            loss_list.append(loss)
            loss_sum = loss_sum + loss
        
        for index, client_data in enumerate(datamodule_list):
            client_ratio_list[index] = loss_list[index]/loss_sum
        client_ratio_list = project_onto_chi_square_ball(client_ratio_list, 1)

        from copy import deepcopy
        print("client_ratio_list:",client_ratio_list)
        server.client_ratio_list = deepcopy(client_ratio_list)
        

        
        ###########################################################################################
        ckpt_path = os.path.join(
            BASE_DIR, f"../../../data/ckpts/FedAvg_FedDRA/server/last.pth")
        torch.save(server.state_dict(), ckpt_path)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Finished parameter averaging!")
        
        #######################################################################################
        # evaluation
        ######################################################################################
        test_result_list = []
        
        for index, datamodule in enumerate(datamodule_list):
            test_result = test_server(args, callbacks_list[index], server, datamodule)[0]
            print(f"finished testing server performance on training data of client{index}!")
            for k,v in test_result.items():
                if type(test_result[k])=="tensor":
                    test_result[k] = test_result[k].to("cpu")
            test_result_list.append(test_result) 
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        
        
        ##########################################################################################
        # logic for metircs
        ##########################################################################################
        

        is_log = 1
        if is_log == 1:
            commu_log = {"commu": commu}
            for index in range(len(model_list)):
            
                commu_log[f"test_loss_on_client{index}"] = test_result_list[index]['test_loss']
                if 'val_loss_ita' in test_result_list[index].keys():
                    commu_log[f"test_val_ita_loss_on_client{index}"] = test_result_list[index]['test_loss_ita']
                if 'val_loss_local' in test_result_list[index].keys():
                    commu_log[f"test_val_local_loss_on_client{index}"] = test_result_list[index]['test_loss_local']
                commu_log[f"test_val_acc1_on_client{index}"] = test_result_list[index]['test_acc1']
                commu_log[f"test_val_acc5_on_client{index}"] = test_result_list[index]['test_acc5']
                commu_log[f"test_diag_cos_mean_of_client{index}"] = test_result_list[index]['diag_cos_mean']
                commu_log[f"test_diag_global_cos_mean_of_client{index}"] = test_result_list[index]['g_diag_cos_mean']
            
            
            
                
            
    
    ######################## after communication #########################################
    print(commu_log)
    
    
if __name__ == "__main__":
    FedPretrain()