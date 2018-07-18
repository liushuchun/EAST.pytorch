import os

from config import opt
import models
import torch  as t
import torch.optim as optim
from data.dataset import ImageDataSet
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from utils.visualize import Visualizer
from utils.log import  Logger
from tqdm import tqdm


def write_csv(results,file_name):
    '''write the result to file'''
    import csv
    with open(file_name,'w') as f:
        writer=csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerow(results)



def train(**kwargs):
    opt.parse(kwargs)

    if not os.path.isdir(opt.log_path):
        os.mkdir(opt.log_path)
    if not os.path.isdir(opt.save_path):
        os.mkdir(opt.save_path)

    # step0:set log
    logger=Logger(opt.log_path)

    # step1:configure model
    model=getattr(models,opt.model)()

    if os.path.exists(opt.load_model_path):
        model.load(opt.load_model_path)

    if opt.use_gpu:model.cuda()

    # step2:data
    train_data=ImageDataSet(opt.train_data_root,train=True)
    val_data=ImageDataSet(opt.train_data_root,train=False)
    train_dataloader=DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    val_dataloader=DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)


    # step3:criterion and optimizer
    criterion=nn.CrossEntropyLoss()
    lr=opt.lr
    if opt.optmizer=="RMSprop":
        optimizer=optim.RMSprop(model.parameters(),lr=lr)
    elif opt.optmizer=="Adam":
        optimizer=optim.Adam(model.parameters(),lr=lr,betas=opt.betas)
    elif opt.optmizer=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=lr)
    else:
        optimizer=optim.Adadelta(model.parameters(),lr=lr)


    for epoch in range(opt.epoch_num):
        for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
            # train model
            input=Variable(data)
            target=Variable(label)

            if opt.use_gpu:
                input=input.cuda()
                target=target.cuda()

            optimizer.zero_grad()
            score=model(input)
            loss=criterion(score,target)

            logger.scalar_summary('train_loss', loss.data[0], ii + epoch * len(train_dataloader))

            accuracy=0
            logger.scalar_summary('train_accuray', accuracy, i + epoch * len(train_dataloader))

            loss.backward()
            optimizer.step()





        model.save()



def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    :param model: 
    :param dataloader: 
    :return: 
    '''
    model.eval()

    for ii,data in enumerate(dataloader):
        input,label=data
        val_input=Variable(input,volatile=True)
        val_label=Variable(label.type(t.LongTensor),volatile=True)
        if opt.use_gpu:
            val_input.cuda()
            val_label.cuda()

        score=model(val_input)


    model.train()




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







if __name__=="__main__":
    import fire
    fire.Fire()