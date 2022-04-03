import torch
import numpy as np

gt = np.random.randint(0,5, size=[15,15])  #先生成一个15*15的label，值在5以内，意思是5类分割任务
gt = torch.LongTensor(gt)



def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


gt_one_hot = get_one_hot(gt, 5)
print(gt_one_hot)
print(gt_one_hot.shape)

print(gt_one_hot.argmax(-1) == gt)  # 判断one hot 转换方式是否正确，全是1就是正确的
