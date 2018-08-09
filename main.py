# coding=utf-8
import os
from loss import *
from config import opt
import models
import torch
import torch.optim as optim
from data.dataset import ImageDataSet, collate_fn
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
# from utils.visualize import Visualizer
# from tqdm import tqdm
from torch.optim import lr_scheduler
import data.dataset
import torch.utils.data as data
import time
import cv2


# gpus = list(range(len(opt.gpu_list.split(','))))


def help():
    '''
    打印帮助信息:python main.py help

    '''
    print('''
    useage:python main.py <function> [--args=value]
    <function>:=train | test | help
    example:
        python {0} train --env='env0701' --lr=0.01
        python {0} test --dataset='pyath/to/dataset/root/'
        python {0} help
    available args:
    '''.format(__file__))
    from inspect import getsource

    source = (getsource(opt.__class__))
    print(source)


def write_csv(results, file_name):
    '''write the result to file'''
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerow(results)


def train(epochs, model, trainloader, crit, optimizer,
          scheduler, save_step, weight_decay):
    for e in range(opt.epoch_num):
        print('*' * 10)
        print('Epoch {} / {}'.format(e + 1, epochs))
        model.train()
        start = time.time()
        loss = 0.0
        total = 0.0
        for i, (img, score_map, geo_map, training_mask) in enumerate(trainloader):
            scheduler.step()
            optimizer.zero_grad()

            img = Variable(img.cuda())
            score_map = Variable(score_map.cuda())
            geo_map = Variable(geo_map.cuda())
            training_mask = Variable(training_mask.cuda())
            f_score, f_geometry = model(img)
            loss1 = crit(score_map, f_score, geo_map, f_geometry, training_mask)

            loss += loss1.data[0]

            loss1.backward()
            optimizer.step()

        during = time.time() - start
        print("Loss : {:.6f}, Time:{:.2f} s ".format(loss / len(trainloader), during))
        print()
        # writer.add_scalar('loss', loss / len(trainloader), e)

        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(e + 1))


def main(**kwargs):
    opt.parse(kwargs)

    # if not os.path.isdir(opt.log_path):
    # os.mkdir(opt.log_path)
    # if not os.path.isdir(opt.save_path):
    # os.mkdir(opt.save_path)

    # step0:set log
    # logger = Logger(opt.log_path)

    # step1:configure model
    model = getattr(models, opt.model)()

    if os.path.exists(opt.load_model_path):
        model.load(opt.load_model_path)

    if opt.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    root_path = 'icdar2017/'
    train_img = root_path + 'images'
    train_txt = root_path + 'labels'
    # print(train_img)
    # print(train_txt)
    trainset = ImageDataSet(train_img, train_txt)
    trainloader = DataLoader(
        trainset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=opt.num_workers)

    crit = LossFunc()
    weight_decay = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # weight_decay=1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000,
                                    gamma=0.94)

    train(epochs=1500, model=model, trainloader=trainloader,
          crit=crit, optimizer=optimizer, scheduler=scheduler,
          save_step=100, weight_decay=weight_decay)

    # write.close()


if __name__ == "__main__":
    main()
