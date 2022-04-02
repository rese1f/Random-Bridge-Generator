import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    
    # experiment
    parser.add_argument('--name', default='test', type=str,
                        help='test name')
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # dataset
    parser.add_argument('--aug', default=True, type=bool,
                        help='if data augmentation')
    
    # model
    parser.add_argument('--model', default='resnet50', type=str)
    
    # train
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--epoch', default=40, type=int)
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args