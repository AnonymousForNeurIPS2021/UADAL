from __future__ import print_function
from data_loader.get_loader import get_dataset_information, get_loader_ss
import random
from utils import utils as utils
from models.basenet_v3 import *

def main(args):

    sum_str = ''
    args_list = [str(arg) for arg in vars(args)]
    args_list.sort()
    for arg in args_list:
        sum_str += '{:>20} : {:<20} \n'.format(arg, getattr(args, arg))
    print(sum_str)

    utils.setGPU(args.set_gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda:%s" % (0))

    source_data, target_data, evaluation_data, num_class, class_list, len_s, len_t = get_dataset_information(args.dataset, args.source_domain, args.target_domain)
    data_transforms = utils.bring_data_transformation(source_data, target_data, evaluation_data, args.dataset, args)
    train_loader, test_loader, test_loader_source, train_loader_ss, train_loader_st = get_loader_ss(args, source_data, target_data, evaluation_data, data_transforms, batch_size=args.batch_size, ss_classes=4, known_classes=num_class-1)
    dataset_train, dataset_test, dataset_test_source = train_loader.load_data(), test_loader, test_loader_source
    dataset_train_ss, dataset_train_st = train_loader_ss, train_loader_st
    args.source_path, args.target_path = source_data, target_data
    from models.model import UAGRL
    model = UAGRL(args, num_class, dataset_train, dataset_test, dataset_test_source, dataset_train_ss, dataset_train_st, len_s, len_t, class_list)
    model.warm_train()
    print('Warmup is finished..')
    model.update_bmm_model()
    print('BMM fitting is finished..')
    model.freeze_GE()
    model.build_main()
    model.main_train()


if __name__ == '__main__':
    from config import args
    main(args)
