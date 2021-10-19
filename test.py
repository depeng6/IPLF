# -*- encoding: utf-8 -*-
"""
@File    : test.py
@Time    : 2021/10/12 15:57
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""

import torch
import numpy as np
from datetime import datetime
import os
from net import Fusion_Net
from config import conf
import dataset
from torchvision import utils as vutils
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

def load_model(path):
    fusion_model = Fusion_Net(1)


    fusion_model = fusion_model.cuda()

    checkpoint = torch.load(path,map_location='cuda:0')

    fusion_model.load_state_dict(checkpoint['model_state_dict'],False)
    fusion_model.eval()
    para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))

    return fusion_model
def main():

    TIME_STAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
    # out_image_path = "Test_result/IPLF_TNO"
    out_image_path = "Test_result/IPLF_CVC"
    mode_path = conf.model_path

    total_time = 0.

    if os.path.exists(out_image_path) is False:
        os.makedirs(out_image_path)


    with torch.no_grad():
        model = load_model(mode_path)
        ir_image_list = dataset.list_images(conf.test_ir_dir)
        vis_image_list = dataset.list_images(conf.test_vi_dir)
        batch_size = 1
        image_set_ir, batches = dataset.load_dataset(ir_image_list, batch_size)
        image_set_vis, batches = dataset.load_dataset(vis_image_list, batch_size)
        for batch in range(batches):

            image_paths_ir = image_set_ir[batch]
            image_paths_vis = image_set_vis[batch]

            img_ir = dataset.get_train_images(image_paths_ir)
            img_vis = dataset.get_train_images(image_paths_vis)


            img_ir = img_ir.cuda()
            img_vis = img_vis.cuda()

            img_ir = Variable(img_ir, requires_grad=False)
            img_vis = Variable(img_vis, requires_grad=False)
            torch.cuda.synchronize()
            start = time.time()
            out_eval = model(img_ir, img_vis)

            torch.cuda.synchronize()
            end = time.time()

            total_time += end-start

            vutils.save_image(out_eval, os.path.join(out_image_path, 'fusion_'f'{batch+1}.png'))
        print(total_time/25)
    print("Done......")





if __name__ == '__main__':
    main()