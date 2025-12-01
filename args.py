import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # basic setting
    parser.add_argument("--name-dataset", type=str, default="mnist",
                        help="dataset for training, 'mnist' or 'cifar' (default: 'mnist')")
    parser.add_argument("--num_clients", type=int, default=100, help="number of clients: K")
    parser.add_argument("--rounds", type=int, default=200, help="rounds of training")
    parser.add_argument("--local_epochs", type=int, default=3, help="the number of local epochs: E")
    parser.add_argument("--batch_size", type=int, default=64, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    
    # byzantine attacker setting
    parser.add_argument("--mal_prop", type=float, default=0, 
                        help="proportion of malicious clients (default: 0)")
    parser.add_argument("--attack_type", type=str, default="label_flip",
                        choices=["none", "label_flip", "gaussian_noise", "sign_flip", "alie", "free_rider"],
                        help="type of byzantine attack")
    parser.add_argument("--alie_eps", type=float, default=0.01, 
                        help="para for ALIE attack")
    parser.add_argument("--alie_scale", type=float, default=0.1, 
                        help="para for ALIE attack: scaling")
    parser.add_argument("--fr_noise", type=float, default=1e-5, 
                        help="para for Free rider attack")

    # aggregation setting
    parser.add_argument("--agg_method", type=str, default="my_algo",
                        choices=["fedavg", "krum", "fltrust","rfa", "esfl", "my_algo"],
                        help="aggregation algorithm")
    
    # Non-IID setting
    parser.add_argument("--iid", action='store_true', help="select to force iid data")
    parser.add_argument("--alpha", type=float, default=0.8, help="dirichlet alpha for non-iid")
    
    # root data for trust bootstrap setting
    parser.add_argument("--root_data_size", type=int, default=100, help="size of root dataset on server")

    # output setting
    parser.add_argument("--out_path", type=str, default="./results", help="path to save results")

    # RFA
    parser.add_argument("--one_step", action='store_true', help="one step for RFA")
    parser.add_argument("--R", type=int, default=3, help="iteration count for RFA")
    parser.add_argument("--rfa_alphas", type=int, default=None, help="alphas for RFA")

    # ESFL: K-Means cluster number
    parser.add_argument("--num_clusters", type=int, default=2,
                        help="Number of clusters for PPRA K-Means (default: 2)")

    args = parser.parse_args()
    return args