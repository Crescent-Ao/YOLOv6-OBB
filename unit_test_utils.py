def prepro_data(batch_data, device):
    images = batch_data[0].float().to(device, non_blocking=True)
    targets = batch_data[1].float().to(device)
    return images, targets
def get_optimizer(args, cfg, model):
    accumulate = max(1, round(64 / args.batch_size))
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64
    optimizer = build_optimizer(cfg, model)
    return optimizer
def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer
def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf
def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        logger.error('unknown lr scheduler, use Cosine defaulted')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
def update_optimizer(step,max_stepnum,epoch,optimizer,args,lf,cfg,scaler,warmup_stepnum,last_opt_step):
    curr_step = step + max_stepnum * epoch
    accumulate = max(1, round(64/args.batch_size))
    
    if curr_step <= warmup_stepnum:
            accumulate = max(1, np.interp(curr_step, [0, warmup_stepnum], [1, 64 / args.batch_size]).round())
            for k, param in enumerate(optimizer.param_groups):
                warmup_bias_lr = cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, warmup_stepnum], [warmup_bias_lr, param['initial_lr'] * lf(epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, warmup_stepnum], [cfg.solver.warmup_momentum, cfg.solver.momentum])
    if curr_step - last_opt_step >= accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            last_opt_step = curr_step
    return last_opt_step
class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count