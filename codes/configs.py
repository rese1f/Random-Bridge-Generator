import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Experience script")

    # experiment
    parser.add_argument("--name", default="test", type=str, help="test name")
    parser.add_argument("--device", default="cuda:0", type=str)

    # dataset
    parser.add_argument(
        "--images-dir",
        default="/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/imgs",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--masks-dir", 
        default="/home/pose3d/projs/UNet_Spine_Proj/UNet_Spine/data/masks", 
        type=str, 
        help="masks path"
    )
    parser.add_argument("--aug", default=True, type=bool, help="if data augmentation")

    # model
    parser.add_argument("--arch", default="Unet", type=str)
    parser.add_argument("--backbone", default="se_resnet50", type=str)

    # train
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--num-epoch", default=50, type=int)
    parser.add_argument("--val-percent",default=0.1,type=float)

    args = parser.parse_args()

    # if configs conflict:
    #   raise Keyerror()

    return args
