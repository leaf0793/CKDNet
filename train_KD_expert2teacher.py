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
def get_model(modes):
    if modes == 'teacher':
        from models.official_model.stackhourglass import PSMNet_KD as teacher_main
        return teacher_main
    if modes == 'teacher2':
        from models.official_model2.stackhourglass import PSMNet_KD_KL as teacher_main
        return teacher_main
    if modes == 'student':
        from models.student1.stackhourglass import PSMNet_KD as student_main
        return student_main
    if modes == 'student2':
        from models.student2.stackhourglass import PSMNet_KD as student_main
        return student_main
    if modes == 'student_short':
        from models.student_short.stackhourglass import PSMNet_KD as student_main
        return student_main
    if modes == 'student_as_teacher':
        from models.student_as_teacher.stackhourglass import PSMNet_KD as student_main
        return student_main
    




def get_teacher_and_student(teacher_name, student_name):
    '''
    '''
    candidate = {'teacher': 'teacher',
                 'teacher2': 'teacher2',
                 'student': 'student',
                 'student2': 'student2',
                 'student_short': 'student_short',
                 'student_as_teacher': 'student_as_teacher',
                 }

    return get_model(candidate[teacher_name]), get_model(candidate[student_name])



def main_worker(gpu, cfg):
    if len(cfg.use_cuda)>1:
        dist.init_process_group(backend = "gloo", world_size=cfg.gpu_num, init_method=cfg.dist_url, rank=gpu)
    if(main_process(gpu)):
        writer = SummaryWriter('runs/exp/{}'.format(date.today()))
    from models.warpers import KD_warper2 as KD_warper4
    # from models.warpers2 import KD_warper3 as KD_warper4
    # from models.warpers3 import KD_warper4

    # teacher, student = get_teacher_and_student('student_as_teacher', 'student')
    # teacher, _ = get_teacher_and_student('student_as_teacher', 'student')
    teacher, student = get_teacher_and_student('teacher', 'student')
    torch.cuda.set_device(gpu)
    # teacher = teacher(192).cuda(gpu)
    teacher = teacher(192, percentage=1.0).cuda(gpu)
    student = student(192, percentage=cfg.slim_pecentage).cuda(gpu)
    # from models_msn.MSNet2D import MSNet2D_KD as student
    student = student(192).cuda(gpu)
    if main_process(gpu):
        print(student)
    
    
    if cfg.loadmodel:
        # teacher, _ = load_model(teacher, None, cfg, gpu)
        teacher, _ = load_model_after_KD(teacher, None, cfg, gpu)
    if cfg.student_loadmodel:
        # cfg.student_loadmodel = cfg.loadmodel
        # cfg.loadmodel = cfg.student_loadmodel 
        cfg.start_epoch = 21
        cfg.max_epoch = 30
        student, _ = load_model_after_KD_student(student, None, cfg, gpu)
    model = KD_warper4(teacher=teacher, student=student, KDlossOnly=True)
    model.cuda(gpu)
    
    if len(cfg.use_cuda)>1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,
                            betas=(0.9, 0.999), weight_decay=1e-2,)
    if cfg.student_loadmodel:
        _, optimizer = load_model_after_KD_student(student, optimizer, cfg, gpu)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=10, verbose=True, threshold=3e-3)


    TrainImgLoader_disp, TestImgLoader_disp = DATASET_disp(cfg)
    small_test_loss = 100
    # for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
    # cfg.start_epoch = 0
    for epoch in (pbar := tqdm(range(cfg.start_epoch, cfg.max_epoch+1))):
        model.Temperature = min(1, 0.5+0.1*(max(epoch, 5)-5))
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
                model.train()
                loss, loss_disp, loss_kd = train(model, data_batch, gpu, optimizer)
                # ! end 
                epoch_loss.append(float(loss))
                if main_process(gpu) and (batch_idx % (len(TrainImgLoader_disp)//10) == 0):
                    message = 'Epoch: {}/{}. Iteration: {}/{}. LR:{:.1e},  Epoch time: {}. Disp loss: {:.3f}. KD loss: {:.3f}. Total loss: {:.3f}. '.format(
                        epoch, cfg.max_epoch, batch_idx, len(TrainImgLoader_disp),
                        float(optimizer.param_groups[0]['lr']), str(datetime.now()-start_time)[:-4],
                        loss_disp, loss_kd, loss)
                    # print(message)
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
    parser.add_argument("--slim_pecentage", default=1, type=float, help="slim_pecentage")
    parser.add_argument("--back", default="", type=str, help="")
    cfg = init_cfg(parser.parse_args())
    # cfg.slim_pecentage = 0.3
        
    cfg.server_name = 'LARGE'
    # cfg.use_cuda = '0,1,2,3'
    cfg.use_cuda = '0,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.use_cuda
    cfg.pad_size= (512, 256)
    
    cfg.want_size = (480, 640)
    cfg.loadmodel = None
    
    cfg.num_class_ratio = 1
    cfg.finetune=None
    cfg.teacher_slim_pecentage = 1.0
    
    # cfg.finetune = 'kitti'
    # cfg.loadmodel = ""
    # cfg.finetune = None
    # cfg.finetune = 'driving'
    # cfg.loadmodel = 'zoo/test_104_0.58898.tar'
    # cfg.loadmodel = 'zoo_zoo/backbone/slim_1.0/0723_31_0.6234.tar'
    # cfg.loadmodel = 'zoo_zoo/kd_nogt_mae/slim_0.8/0803_0020_0.6754.tar'
    cfg.loadmodel = 'zoo_zoo/kd_nogt_mae/slim_0.8/0803_0036_0.6431.tar'
    # cfg.loadmodel = 'zoo_zoo/kd_nogt_mae/slim_0.3/0805_0036_0.8885.tar'
    # cfg.student_loadmodel = 'zoo_zoo/KD_by_nogt_mae_0.5_MSN/slim_1/0331_0020_1.0361.tar'
    # cfg.loadmodel = 'zoo_zoo/kd_nogt_mae/slim_0.5/0804_0036_0.6966.tar'
    # cfg.loadmodel = 'zoo_zoo/kd_nogt_mae/slim_0.5/0804_0036_0.6966.tar'
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
    
    # file_dir = "zoo_zoo/KD_feature/{}".format(f"slim_{cfg.slim_pecentage}")
    
    file_dir = "zoo_zoo/KD_feature/{}".format(f"slim_{cfg.slim_pecentage}")
    file_dir = f"zoo_zoo/KD_by_nogt_mae_{cfg.teacher_slim_pecentage}_MSN/slim_{cfg.slim_pecentage}"
    if cfg.back != "":
        file_dir += "_"+cfg.back
    # file_dir = "./zoo_zoo/kd_mae/{}".format(f"slim_{cfg.slim_pecentage}")
    file_dir_0 = ''
    for files in file_dir.split('/'):
        file_dir_0 +=files +'/'
        if not os.path.exists(file_dir_0):
            os.mkdir(file_dir_0)
    del file_dir_0
    # for files in range(len(file_dir.split(''))):
        
    from datetime import datetime
    current = datetime.now()
    cfg.save_prefix = "{}/{:02}{:02}".format(file_dir, current.month, current.day)

    cfg = get_cfg(cfg)
    cfg.max_epoch=30
    cfg.start_epoch = 0
    cfg.disp_batch = 6
    if len(cfg.use_cuda) > 1:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=cfg.gpu_num,
                 args=(cfg,))
    else:
        main_worker(int(cfg.use_cuda), cfg)

