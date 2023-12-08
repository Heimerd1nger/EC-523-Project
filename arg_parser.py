import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Unleaning")

##################################### Dataset #################################################
    parser.add_argument(
        "--bs", type=str, default=128, help="data loader batch size"
    )


##################################### Unlearner ############################################
    # influence func - woodfisher
    # parser.add_argument("--alpha_wf", default=0.2, type=float, help="unlearn noise")




    # SCRUB
    parser.add_argument('--sgda_epochs', type=int, default=10, help='Number of epochs for SGDA')
    parser.add_argument('--sgda_learning_rate', type=float, default=0.005, help='Learning rate for SGDA')
    # parser.add_argument('--r_sgda_learning_rate', type=float, default=0.005, help='Learning rate for SGDA')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[3, 5, 9], help='Epochs at which to decay learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Decay rate for learning rate')
    parser.add_argument('--sgda_weight_decay', type=float, default=5e-4, help='Weight decay for SGDA optimizer')
    parser.add_argument('--sgda_momentum', type=float, default=0.9, help='Momentum for SGDA optimizer')
    parser.add_argument('--kd_T', type=float, default=2, help='Temperature for knowledge distillation')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma parameter for some algorithm')
    parser.add_argument('--alpha', type=float, default=0.001, help='Alpha parameter for some algorithm')
    parser.add_argument('--smoothing', type=float, default=0.0, help='Smoothing parameter')
    parser.add_argument('--msteps', type=int, default=10, help='Number of steps for something')
    parser.add_argument('--sub_sample', type=int, default=0, help='Number of subsampling on retain set')
    parser.add_argument('--checkpoints', action='store_true', help='Whether to load the checkpoint or not')



    return parser.parse_args()
