import warnings
class DefaultConfig(object):
    train_task_id = "2T736"

    initial_epoch = 0
    epoch_num = 24
    lr = 1e-3
    decay = 5e-4
    use_gpu=True
    batch_size=128
    num_workers=8
    optmizer='RMSprop' # RMSprop,Adam # SGD, SGD
    betas=(0.5,0.999)
    epsilon=1e-4
    shrink_side_ratio=0.6
    shrink_ratio=0.2
    model='EAST'



    patience = 2
    load_weights = False
    lambda_inside_score_loss = 4.0
    lambda_side_vertex_code_loss = 1.0
    lambda_side_vertex_coord_loss = 1.0

    train_data_root = "/data/train/"
    val_data_root="/data/val/"
    test_data_root="/data/test/"
    log_path="./checkpoints"
    load_model_path="checkpoints/model.pth"
    save_path="./model"
    debug_file="/tmp/debug"
    result_file='result.csv'

    gpu_list = "0,1,2,3,4,5,6,7"


    total_img = 10000
    validation_split_ratio = 0.1
    max_train_img_size = 736
    max_predict_img_size = 2400

    assert max_train_img_size in [256, 384, 512, 640, 736], 'max_train_img_size must in [256~736]'

    if max_train_img_size == 256:
        batch_size = 8
    elif max_train_img_size == 384:
        batch_size = 4
    elif max_train_img_size == 512:
        batch_size = 2

    else:
        batch_size = 1

    steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size

    validation_steps = total_img * validation_split_ratio // batch_size




def parse(self,kwargs):
    '''
        update the config params
    :param self: 
    :param kwargs: 
    :return: 
    '''
    for k,v in kwargs.iteritems():
        if not hasattr(self,k):
            warnings.warn("Warning:opt has not attribute ^s" %k)

        setattr(self,k,v)

    print('use config:')
    for k,v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k,getattr(self,k))


DefaultConfig.parse=parse
opt=DefaultConfig()