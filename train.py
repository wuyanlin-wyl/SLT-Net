import torch as t
import torch.nn as nn
import torch.nn.functional as F
import cfg
import numpy as np
import os
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import Dataset_train_test
from evalution_segmentation1 import eval_semantic_segmentation
# from Unet import UNet
from litttle_Unet4 import UNet
# from xiaorong10 import UNet
# from xiaorong import UNet
# from net.Unet_nest import NestedUNet
# from net.segnet_4down import SegNet
# from net.segnet_4down import SegNet
# from net.UNet_3Plus import UNet_3Plus as UNet
# from net.cenet import CE_Net_ as UNet
# from net.core.res_unet import ResUnet as UNet
# from net.core.res_unet_plus import ResUnetPlusPlus as UNet
# from Transformer.FAT_512 import FAT_Net as UNet

from utils.保存训练数据为csv文件 import data_write_csv
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")
np.seterr(divide='ignore', invalid='ignore')

# -----设置GPU运行个数------
# gpu_list = [0, 1]
# gpu_list_str = ','.join(map(str, gpu_list))
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print("\033[34m当前计算设备为{}\033[0m".format(device))

Cam_train = Dataset_train_test([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.train_crop_size)
Cam_val = Dataset_train_test([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.val_crop_size)

train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=6)
val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=6)

unet = UNet(3,2)
# unet = UNet(3)

# -----GPU并行-----
# unet = nn.DataParallel(unet)

unet = unet.to(device)
criterion = nn.NLLLoss().to(device)
# optimizer = optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-8)#正则化，权重衰减
optimizer = optim.Adam(unet.parameters(), lr=1e-4)


#------- 注册hook---------
# fmap_dict = {'conv':[]}
#
# def hook_func(m, i, o):
#     fmap_dict['conv'].append(o)
#
# unet.up1.conv.double_conv[0].register_forward_hook(hook_func)
# # unet.up2.conv.double_conv[0].register_forward_hook(hook_func)
# # unet.up3.conv.double_conv[0].register_forward_hook(hook_func)
# unet.up4.conv.double_conv[0].register_forward_hook(hook_func)

def train(model):
    best = [0]
    best_epoch = 0
    iter_train = 0
    iter_val = 0
    train_lossData = [[]]
    train_accData = [[]]
    val_lossData = [[]]
    val_accData = [[]]

    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print("开始第{}轮模型训练".format(epoch+1))
        # print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 3 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
                print('当前学习率为：',group['lr'])
        # 指标初始化
        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        total = 0

        net = model.train()
        # 训练批次
        train_time = datetime.now()
        train_bar = tqdm(train_data,colour='blue')
        for i, sample in enumerate(train_bar):
            iter_train += 1
            labels = sample['label']
            total += labels.size(0)
            # 载入数据
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                         cfg.EPOCH_NUMBER,
                                                         loss.item())

            # train_lossData.append([iter_train, loss.cpu().data.numpy()])  # 先转成普通tensor，再转成numpy形式
            train_lossData.append([iter_train, train_loss/iter_train])  # 先转成普通tensor，再转成numpy形式

            # --------------add image可视化特征图--------------
            # fmap = fmap_dict['conv']
            # fmap = t.stack(fmap)
            # fmap.squeeze_(0)
            # print(fmap.shape)
            #
            # fmap.transpose_(0, 1)
            # print(fmap.shape)
            #
            # nrow = int(np.sqrt(fmap.shape[0]))
            # fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
            # writer.add_image('feature map in conv1', fmap_grid, global_step=322)

            # 评估
            preout = out.max(dim=1)[1].data.cpu().numpy()
            gtout = img_label.data.cpu().numpy()

            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            train_accData.append([iter_train, train_acc / iter_train])

            # --------记录数据，保存于event file，tensorboard可视化训练过程----------
            #  writer.add_scalars("Loss", {"Train": loss.item()}, iter_train)
            #  writer.add_scalars("Accuracy", {"Train": train_acc / total }, iter_train)

            # print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))

        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data)
        )
        print(metric_description)

        # ------------训练推理时间---------------
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - train_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(time_str)

        # ---------每个epoch，记录梯度，权值，用于tensorboard可视化观察梯度-----------
        # for name, param in net.named_parameters():
        #     # writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)

        # 验证网络模型
        if (epoch + 1) % 1 == 0:
            print("开始第{}轮模型验证".format(epoch + 1))
            net = model.eval()

            # 参数初始化
            eval_loss = 0
            eval_acc = 0
            eval_miou = 0
            eval_class_acc = 0
            total_val = 0

            val_time = datetime.now()
            val_bar = tqdm(val_data, colour='red')
            with t.no_grad():
                for j, sample in enumerate(val_bar):
                    iter_val += 1
                    labels_ = sample['label']
                    total_val += labels_.size(0)

                    valImg = Variable(sample['img'].to(device))
                    valLabel = Variable(sample['label'].long().to(device))

                    out = net(valImg)
                    out = F.log_softmax(out, dim=1)
                    val_loss = criterion(out, valLabel)
                    eval_loss = val_loss.item() + eval_loss

                    val_bar.desc = "val iteration[{}/{}] loss:{:.3f}".format(j + 1,
                                                                             len(val_data),
                                                                             val_loss.item())

                    preout = out.max(dim=1)[1].data.cpu().numpy()
                    gtout = valLabel.data.cpu().numpy()

                    pre_label = out.max(dim=1)[1].data.cpu().numpy()
                    pre_label = [i for i in pre_label]

                    true_label = valLabel.data.cpu().numpy()
                    true_label = [i for i in true_label]

                    eval_metrics = eval_semantic_segmentation(pre_label, true_label, preout, gtout)
                    eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
                    eval_miou = eval_metrics['miou'] + eval_miou
                    eval_class_acc = eval_metrics['class_accuracy'] + eval_class_acc

                val_lossData.append([epoch, eval_loss / len(val_data)])
                val_accData.append([epoch, eval_acc / len(val_data)])

                # ---------记录数据，保存于event file，用于tensorboard可视化验证过程-------------
                # writer.add_scalars("Loss", {"Valid": loss1.item() }, iter_val)
                # writer.add_scalars("Accuracy", {"Valid": eval_acc / total_val}, iter_val)

                val_str = (
                    '|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
                        eval_loss / len(val_data),
                        eval_acc / len(val_data),
                        eval_miou / len(val_data),
                        eval_class_acc / len(val_data)))
                print(val_str)

                # ------------验证过程推理时间---------------
                cur_time = datetime.now()
                h, remainder = divmod((cur_time - val_time).seconds, 3600)
                m, s = divmod(remainder, 60)
                time_str = 'Val_Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
                print(time_str)

                # ------------保存权重文件---------------
                if max(best) <= eval_miou / len(val_data):
                    best.append(eval_miou / len(val_data))
                    t.save(net.state_dict(), './weight/{}.pth'.format(epoch + 1))
                    best_epoch = epoch + 1
                print("当前模型最大IOU为{:.5f},对应epoch次数为{}".format(best[-1], best_epoch))

        # 服务器
        data_write_csv("./curve/train_loss.csv", train_lossData)
        data_write_csv("./curve/train_acc.csv", train_accData)
        data_write_csv("./curve/val_loss.csv", val_lossData)
        data_write_csv("./curve/val_acc.csv", val_accData)
        print("\033[34m网络数据保存成功\033[0m")
        print('*'*10,'分隔符','*'*10)



        # ----------tensorboard可视化指标------------
        # log_dir = os.path.join('tensorboard', 'train', 'Accuracy')
        # train_writer = SummaryWriter(log_dir=log_dir)
        # train_writer.add_scalar('Accuracy', train_acc/len(train_data), epoch)
        #
        # log_dir = os.path.join('tensorboard', 'train', 'MIOU')
        # train_writer = SummaryWriter(log_dir=log_dir)
        # train_writer.add_scalar('MIOU', train_miou / len(train_data), epoch)


if __name__ == "__main__":
    train(unet)
