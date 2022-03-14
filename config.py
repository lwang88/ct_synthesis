import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--data_type', type=str, default='direct')

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--depth', type=int, default=3, help='# of pooling')
model_arg.add_argument('--nChannel', type=int, default=1)
model_arg.add_argument('--n_resblocks', type=int, default=12)
model_arg.add_argument('--up_mode', type=str, default='shuffle')
model_arg.add_argument('--upscale', type=int, default=2, help='scale_factor')
model_arg.add_argument("--resume", action='store_true', help='run resume or not')
 
# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train')
learn_arg.add_argument('--loss', type=str, default='L1')
learn_arg.add_argument('--lr', type=float, default=(1e-4))
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--batch_size', type=int, default=1)
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=200)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--exp_name', type=str, default='exp')
misc_arg.add_argument('--log_iter', type=int, default=20)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--num_workers', type=int, default=8)

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed
