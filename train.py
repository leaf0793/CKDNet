import numpy as np
from models import *

# from datasets import 
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from datetime import datetime, date
import os
from utils.train_util import *
# from tqdm import tqdm
from tqdm import tqdm, trange


os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'


def main_worker(gpu, cfg):
    if len(cfg.use_cuda)>1:
        dist.init_process_group(backend = "gloo", world_size=cfg.gpu_num, init_method=cfg.dist_url, rank=gpu)
    if(main_process(gpu)):
        writer = SummaryWriter('runs/exp/{}'.format(date.today()))
    from models.warpers import Loss_warper
    # from models.teacher.modules_save import teacher_main
    # from models.student.modules import student_main
    # from models.naive.modules import naive_main
    # from models.rk.modules import rk_main
    # from models.student1.stackhourglass import PSMNet
    from models.official_model.stackhourglass import PSMNet
    torch.cuda.set_device(gpu)
    model = PSMNet(192).cuda(gpu)
    # model = PSMNet(192, percentage=cfg.slim_pecentage).cuda(gpu)
    if main_process(gpu):
        print(model)
    if cfg.loadmodel:
        model, _ = load_model(model, None, cfg, gpu)
    model = Loss_warper(model).cuda(gpu)
    if len(cfg.use_cuda)>1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,
                            betas=(0.9, 0.999), weight_decay=1e-2,)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=10, verbose=True, threshold=3e-3)
    
    # model, optimizer = load_model(model, optimizer, cfg, gpu)
        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)
    TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)
    small_test_loss = 100
    # for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
    for epoch in (pbar := tqdm(range(cfg.start_epoch, cfg.max_epoch+1))):
        adjust_learning_rate(cfg=cfg,optimizer=optimizer, epoch=epoch)
        TrainImgLoader_disp.sampler.set_epoch(epoch)
        epoch_loss = []
        start_time = datetime.now()
        if 1:
            for batch_idx, data_batch in enumerate(TrainImgLoader_disp):
                # if (batch_idx > len(TrainImgLoader_disp)*(epoch-10)/40 and epoch < 50) or \
                #         (batch_idx > (len(TrainImgLoader_disp)/10) and epoch <= 10):
                #     break
                
                # ! step 1: train disp
                
                loss, loss_disp, loss_head = train(model, data_batch, gpu, optimizer)
                # ! end 
                epoch_loss.append(float(loss))
                if main_process(gpu) and (batch_idx % (len(TrainImgLoader_disp)//10) == 0):
                    message = 'Epoch: {}/{}. Iteration: {}/{}. LR:{:.1e},  Epoch time: {}. Disp loss: {:.3f}. '.format(
                        epoch, cfg.max_epoch, batch_idx, len(TrainImgLoader_disp),                    
                        float(optimizer.param_groups[0]['lr']), str(datetime.now()-start_time)[:-4],
                        loss)

                    pbar.set_description(f"{message}")
                # step = batch_idx+epoch*len(TrainImgLoader_disp)
                # writer.add_text('train/record', message, epoch)
                # writer.add_scalar('train/Loss', loss, step)
                # writer.add_scalar('train/Disp_loss',  loss_disp, step)
                # writer.add_scalar('train/Head_loss',  loss_head, step)

        # ! -------------------eval-------------------
        if eval_epoch(epoch,cfg=cfg):
            loss_all = []
            start_time = datetime.now()
            for _, data_batch in enumerate(TestImgLoader_disp):
                with torch.no_grad():
                    # losses = test(model, data_batch, gpu)
                    model.eval()
                    disp_loss, _ = test(model, data_batch, gpu)
                    # if cfg.head_only:
                    #     loss_all.append(head_loss)
                    # else:
                    loss_all.append(disp_loss)
            loss_all = np.mean(loss_all, 0)
            loss_all = Error_broadcast(loss_all,len(cfg.use_cuda.split(',')))[0]
            # scheduler.step(loss_all)
            if main_process(gpu):
                writer.add_scalar('full test/Loss', loss_all, epoch)
                message = 'Epoch: {}/{}. Epoch time: {}. Eval Disp loss: {:.3f}. '.format(
                    epoch, cfg.max_epoch, str(datetime.now()-start_time)[:-4],
                    loss_all)
                print(message)

                small_test_loss = loss_all
                save_model_dict(epoch, model.state_dict(),
                                    optimizer.state_dict(), loss_all,cfg)

if __name__ == '__main__':
    import argparse
    from utils.common import init_cfg, get_cfg
    parser = argparse.ArgumentParser(description='PSMNet')
    cfg = init_cfg(parser.parse_args())
        
    cfg.server_name = 'LARGE'
    cfg.use_cuda = '0,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg.pad_size= (512, 256)
    
    cfg.want_size = (480, 640)
    cfg.loadmodel = None
    
    cfg.num_class_ratio = 1
    cfg.finetune=None
    
    # cfg.finetune = 'kitti'
    # cfg.loadmodel = ""
    # cfg.finetune = None
    # cfg.finetune = 'driving'
    # cfg.loadmodel = 'zoo/test_104_0.58898.tar'
    # cfg.loadmodel = 'official_zoo/pretrainedsceneflow_100_00.tar'
    # cfg.loadmodel = 'zoo_rk_new/rk_scene_210_2.45909.tar'
    # cfg.loadmodel = '/home/pan/Works/code/multitask/zoo/test_7.tar'
    # cfg.loadmodel = 'zoo/test_71_0.61438.tar'
    # cfg.finetune = None
    # cfg.loadmodel = 'zoo/head_newplace_46_0.69635.tar'
    # cfg.loadmodel = 'zoo_best/test_noheadloss_naive_driving_56_0.62600.tar'
    # cfg.loadmodel = 'zoo_rk/rk_60_2.31587.tar'
    # cfg.start_epoch = 0 if cfg.loadmodel is None else int(
    #     cfg.loadmodel.split('_')[1][:-4])+1
    # cfg.save_prefix = "./zoo_naive_kitti12/{}".format("naive_only_kitti12")
    cfg.slim_pecentage = 1.0
    file_dir = "./zoo_zoo/backbone/{}".format(f"slim_{cfg.slim_pecentage}")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    from datetime import datetime
    current = datetime.now()
    cfg.save_prefix = "{}/{:02}{:02}".format(file_dir, current.month, current.day)
    cfg = get_cfg(cfg)
    cfg.disp_batch = 10
    if len(cfg.use_cuda) > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=cfg.gpu_num,
                 args=(cfg,))
    else:
        main_worker(int(cfg.use_cuda), cfg)

