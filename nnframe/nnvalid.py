import torch
import torch.nn as nn
from .utils import AvgrageMeter


def valid(
    net,
    criterion,
    epoch,
    valid_loader,
    device='cuda',   
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
    net.eval()
    batch_size = valid_loader.batch_size
    dataset_length = dataset_length_function(valid_loader)
    eval_ans = {}
    eval_item = {}
    for f in eval_function.keys():
        eval_ans[f] = 0.0
        eval_item[f] = AvgrageMeter()

    with(torch.no_grad())
        for idx, (x, t) in enumerate(valid_loader):
            output = net(x.to(device))
            loss = criterion(output, t)
            for f in eval_function.keys():
                eval_item[f].update(*eval_function[f][0](x, output, t))
            if report_freq is not None and idx % report_freq == 0:
                if logging is not None:
                    logging.info(
                        logging_header +
                        f'valid epoch: {epoch}, processed: {idx * batch_size}/{dataset_length} =  {idx * batch_size/dataset_length}'
                    )

            if tbwriter is not None:
                for f in eval_function.keys():
                    tbwriter.add_scalars(eval_function[f][1],
                                        {f, eval_item[f].avg}, epoch)

    if logging is not None:
        string = logging_header + f'valid epoch: {epoch}, '
        for f in eval_function.keys():
            string += f + str(eval_item[f].avg)
        logging.info(string)
