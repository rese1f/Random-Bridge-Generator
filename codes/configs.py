import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    
    # experiment
    parser.add_argument('--name', default='test', type=str,
                        help='test name')
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # dataset
    parser.add_argument('--root-dir', default='C:/Users/Reself/Downloads/Tokaido/', type=str,
                        help='dataset path')
    parser.add_argument('--map-dir', default='C:/Users/Reself/Downloads/Tokaido/files_train.csv', type=str,
                        help='index file path')
    parser.add_argument('--aug', default=True, type=bool,
                        help='if data augmentation')
    
    # model
    parser.add_argument('--model', default='resnet50', type=str)
    
    # train
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=5, type=int)
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args