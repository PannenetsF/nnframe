import torch
import torch.nn as nn
from .utils import AvgrageMeter


def train(
    net,
    optimizer,
    criterion,
    epoch,
    train_loader,
    device='cuda',  
    lr_scheduler=None,  
    grad_clip=None,  
    tbwriter=None,  
    logging=None,  
    logging_header=None,  
    report_freq=50,  
    dataset_length_function=lambda loader: len(loader.dataset),  
    sub_process_in_loop=lambda: None, 
    sub_process_out_loop=lambda: None,
    eval_function={
        'loss': [lambda x, out, t, loss: (loss, 1), 'tb_group'],
        'top1': [lambda x, out, t, loss: (0.1, 1), 'tb_group'],
        'top5': [lambda x, out, t, loss: (0.1, 1), 'tb_group'],
        'iou': [lambda x, out, t, loss: (0.1, 1), 'tb_group'],
        'map': [lambda x, out, t, loss: (0.1, 1), 'tb_group']
    }):
    net.train()
    batch_size = train_loader.batch_size
    dataset_length = dataset_length_function(train_loader)
    eval_ans = {}
    eval_item = {}
    for f in eval_function.keys():
        eval_ans[f] = 0.0
        eval_item[f] = AvgrageMeter()

    for idx, (x, t) in enumerate(train_loader):
        output = net(x.to(device))
        optimizer.zero_grad(0)
        loss = criterion(output, t)
        loss.backward()
        if grad_clip is not None:
            parameters = []
            for group in optimizer.param_groups:
                parameters.append(group['param'])
            nn.utils.clip_grad_norm(parameters, grad_clip)
        optimizer.step()
        for f in eval_function.keys():
            eval_item[f].update(*eval_function[f][0](x, output, t))
        if report_freq is not None and idx % report_freq == 0:
            if logging is not None:
                logging.info(
                    logging_header +
                    f'train epoch: {epoch}, processed: {idx * batch_size}/{dataset_length} =  {idx * batch_size/dataset_length}'
                )

        if tbwriter is not None:
            for f in eval_function.keys():
                tbwriter.add_scalars(eval_function[f][1],
                                     {f, eval_item[f].avg}, epoch)

    if logging is not None:
        string = logging_header + f'train epoch: {epoch}, '
        for f in eval_function.keys():
            string += f + str(eval_item[f].avg)
        logging.info(string)

    if lr_scheduler is not None:
        lr_scheduler.step()
