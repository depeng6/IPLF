# -*- encoding: utf-8 -*-
"""
@File    : config.py
@Time    : 2021/9/23 15:08
@Author  : Chool
@Email   : depeng_cust@163.com
@Software: PyCharm
"""


class conf():

    mult_device = False
    model_name = "train1"

    epochs = 50 #"number of training epochs, default is 2"
    if mult_device:
        batch_size = 64 #"batch size for training, default is 4"
        batch_size_eval = 1
        batch_size_test = 1

        nrows = 16
    else:
        batch_size = 24
        nrows = 8
        batch_size_eval = 1
        batch_size_test = 1

    lr = 1e-4
    log_interval = 10

    image_path = r"./dataset\cvc_data\ir/image_ir_vis"

    save_model_dir = r"checkpoint/"

    model_path = r'./checkpoint/IPLF_model_param.pth'

    test_ir_dir = r"./Test_image/CVC/CVC_IR"
    test_vi_dir = r"./Test_image/CVC/CVC_VIS"

    # test_ir_dir = r"./Test_image/TNO/New_Test_ir"
    # test_vi_dir = r"./Test_image/TNO/New_Test_vi"