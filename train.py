# -*- encoding: utf-8 -*-
"""
@File    : train.py
@Time    : 2021/9/23 14:43
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""
import os
from tqdm import trange
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from datetime import datetime
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import time
from config import conf
from dataset import Fusion_Dataset
import dataset
from net import Fusion_Net
from loss import ssim,image_edge

if conf.mult_device:
    device_ids = [0, 1]
else:
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get lr
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(writer_val):

    model_path = os.path.join(conf.save_model_dir, conf.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)


    fusion_dataset = Fusion_Dataset(conf.image_path)
    fusion_dataloader = DataLoader(fusion_dataset,conf.batch_size,shuffle=True)
    num_samples = len(fusion_dataloader)
    print('Train images samples %d.' % num_samples)

    fusion_model = Fusion_Net(out_channels=1)

    if conf.mult_device:
        fusion_model = torch.nn.DataParallel(fusion_model, device_ids=device_ids)
        fusion_model = fusion_model.cuda(device=device_ids[0])
    else:
        fusion_model = fusion_model.cuda()

    optimizer = optim.Adam(fusion_model.parameters(), conf.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    l1_loss = torch.nn.L1Loss()

    tbar = trange(conf.epochs)
    print('Start training.....')

    best_loss = 100.
    for e in range(0,conf.epochs):
        print('\r Epoch %d.....' % (e + 1))

        all_ssim_loss_gt = 0.
        all_L1_loss = 0.
        all_total_loss = 0.
        all_gra_loss = 0.

        for iteration, batch in enumerate(fusion_dataloader):
            img_ir, img_vis, img_gt = batch

            if conf.mult_device:
                img_ir = img_ir.cuda(device=device_ids[0])
                img_vis = img_vis.cuda(device=device_ids[0])
                img_gt = img_gt.cuda(device=device_ids[0])
            else:
                img_ir = img_ir.cuda()
                img_vis = img_vis.cuda()
                img_gt = img_gt.cuda()

            optimizer.zero_grad()

            outputs = fusion_model(img_ir, img_vis)

            img_edge_gt = image_edge(img_gt)
            img_edge_out = image_edge(outputs)

            _,_,_,s_ir = ssim(outputs, img_gt)
            L1_loss = l1_loss(img_edge_out, img_edge_gt)

            ssim_ir = 1-s_ir

            ssim_ir /= len(outputs)
            L1_loss /= len(outputs)

            total_loss = 70 * ssim_ir + 30 * L1_loss
            total_loss.backward()
            optimizer.step()

            all_ssim_loss_gt += ssim_ir.item()
            all_total_loss += total_loss.item()
            all_L1_loss += L1_loss.item()

            if (iteration + 1) % conf.log_interval == 0:

                mesg = "{} Epoch{}: [{}/{}] L1: {:.6f} ssim: {:.6f} total: {:.6f} lr:{:.6f}".format(
                    time.ctime(), (e + 1), iteration, num_samples,
                    all_L1_loss / conf.log_interval,
                    all_ssim_loss_gt / conf.log_interval,
                    all_total_loss / conf.log_interval,
                    get_lr(optimizer)
                )
                tbar.set_description(mesg)
                if (all_total_loss / conf.log_interval < best_loss) and e != 0:
                    best_loss = all_total_loss / conf.log_interval

                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': fusion_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    save_model_filename = model_path + "/" + conf.model_name + "epoch_" + str(e + 1) + "_" + \
                                          "loss_" + str(round(best_loss, 6)) + \
                                          str(time.ctime()).replace(" ", "_").replace(":", "_") + "_plk"
                    torch.save(checkpoint, save_model_filename)

                    print("\n Validation...")
                    time.sleep(0.5)
                    del img_ir, img_vis, img_gt, outputs,  total_loss, \
                        all_ssim_loss_gt, all_L1_loss, all_total_loss,all_gra_loss

                    eval(fusion_model, e, writer_val)
                    fusion_model.train()

                all_ssim_loss_gt = 0.
                all_total_loss = 0.
                all_L1_loss = 0.
                all_gra_loss = 0.

        lr_scheduler.step()  # update learning rate


def eval(fusion_model, e, writer_val):
    torch.cuda.empty_cache()
    with torch.no_grad():
        fusion_model.eval()
        ir_image_list = dataset.list_images(conf.test_ir_dir)
        vis_image_list = dataset.list_images(conf.test_vi_dir)
        batch_size = conf.batch_size_eval
        image_set_ir, batches = dataset.load_dataset(ir_image_list, batch_size)
        image_set_vis, _ = dataset.load_dataset(vis_image_list, batch_size)
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vis = image_set_vis[batch * batch_size:(batch * batch_size + batch_size)]
            img_ir = dataset.get_train_images(image_paths_ir)
            img_vis = dataset.get_train_images(image_paths_vis)

            if conf.mult_device:
                img_ir = img_ir.cuda(device=device_ids[0])
                img_vis = img_vis.cuda(device=device_ids[0])
            else:
                img_ir = img_ir.cuda(0)
                img_vis = img_vis.cuda(0)

            out_eval = fusion_model(img_ir, img_vis)
            dataset.save_valid_image(out_eval, e, batch)

    del img_ir, img_vis, out_eval
    torch.cuda.empty_cache()


if __name__ == '__main__':
    TIME_STAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    train_log_dir = "./logs/tensorboard/train" + conf.model_name + "_" + TIME_STAMP
    val_log_dir = "./logs/tensorboard/valid" + conf.model_name + "_" + TIME_STAMP

    if os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if os.path.exists(val_log_dir):
        os.makedirs(val_log_dir)
    writer_train = SummaryWriter(log_dir=train_log_dir)
    writer_val = SummaryWriter(log_dir=val_log_dir)
    writer_val.close()

    train(writer_val)





















