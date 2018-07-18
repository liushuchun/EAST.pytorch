import torch as t
import torch.nn as nn
import config as cfg

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x





def reduce_min(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.min(a, keepdim=keepdim)[0]
    return x


def reduce_max(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.max(a, keepdim=keepdim)[0]
    return x


def torch_arctanh(x, eps=1e-6):
    x *= (1. - eps)
    return (t.log((1 + x) / (1 - x))) * 0.5


def l2r_dist(x, y, keepdim=True, eps=1e-8):
    d = (x - y)**2
    d = reduce_sum(d, keepdim=keepdim)
    d += eps  # to prevent infinite gradient at 0
    return d.sqrt()


def l2_dist(x, y, keepdim=True):
    d = (x - y)**2
    return reduce_sum(d, keepdim=keepdim)


def l1_dist(x, y, keepdim=True):
    d = t.abs(x - y)
    return reduce_sum(d, keepdim=keepdim)


def l2_norm(x, keepdim=True):
    norm = reduce_sum(x*x, keepdim=keepdim)
    return norm.sqrt()


def l1_norm(x, keepdim=True):
    return reduce_sum(x.abs(), keepdim=keepdim)


def rescale(x, x_min=-1., x_max=1.):
    return x * (x_max - x_min) + x_min


def tanh_rescale(x, x_min=-1., x_max=1.):
    return (t.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def cross_quad_loss(predict,label,cfg:cfg.DefaultConfig):

    # loss for inside_score
    logits=predict[:,:,:,:1]
    labels=label[:,:,:,:1]
    #balance positive and negative samples in an image
    beta=1-t.mean(labels)
    #first appl sigmoid activation
    predicts=t.sigmoid(logits)

    inside_score_loss=t.mean(-1*(beta*labels*p.log(predicts+cfg.epsilon)+(1-beta)*(1-labels+cfg.epsilon)))

    inside_score_loss*=cfg.lambda_inside_score_loss



    # loss for side_vertex_code
    vertex_logits=predict[:,:,:,1:3]
    vertex_labels=label[:,:,:,1:3]
    vertex_beta=1-(t.mean(predict[:,:,:,1:2])/t.mean(labels)+cfg.epsilon)

    vertex_predicts=t.sigmoid(vertex_logits)
    pos=-1*vertex_beta*vertex_labels*t.log(vertex_predicts+cfg.epsilon)
    neg=-1*(1-vertex_beta)*(1-vertex_labels)*t.log(1-vertex_predicts+cfg.epsilon)

    positive_weights=predict[:,:,:,0].eq(1).float()


    side_vertex_code_loss=reduce_sum(reduce_sum(pos+neg),axis=-1)*positive_weights/(reduce_sum(positive_weights)+cfg.epsilon)
    side_vertex_code_loss*=cfg.lambda_side_vertex_code_loss

    #loss for side_vertex_coord delta
    g_hat=predict[:,:,:,3:]
    g_label=label[:,:,:,3:]

    vertex_weights=predict[:,:,:,1].eq(1).float()

    smooth_l1_loss_fn=t.nn.SmoothL1Loss()
    pixel_wise_smooth_l1norm=smooth_l1_loss_fn(g_hat,g_label)

    side_vertex_coord_loss=reduce_sum(pixel_wise_smooth_l1norm)/(reduce_sum(vertex_weights)+cfg.epsilon)

    side_vertex_coord_loss*=cfg.lambda_side_vertex_coord_loss

    return inside_score_loss+side_vertex_code_loss+side_vertex_coord_loss