import torch
import torch.nn as nn
from torch.autograd.function import Function

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)) # 初始化所有类别的中心
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim # embedding dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1) # tensor([batch_size])
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size) # ctx保留forward参数供backward使用
        centers_batch = centers.index_select(0, label.long()) # 选出当前batch中所有的类别的中心
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size # 返回当前batch的embedding与对应的中心的距离

    @staticmethod
    def backward(ctx, grad_output): # grad_output是forward(input)的输出关于输出的导数，多数情况下是1，为了符合链式法则而已
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long()) # 挑选出batch中类别的center，可重复
        diff = centers_batch - feature # 求出当前center和batch中数据的差值
        # init every iteration
        counts = centers.new_ones(centers.size(0)) # 得到长度为batch_size的全1向量
        ones = centers.new_ones(label.size(0)) # 得到
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones) # 对于当前batch，对每个类别的sample个数计数
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff) # 累计每个类别相应的差值
        grad_centers = grad_centers/counts.view(-1, 1) # 求出最后的导数
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None # 分别是对于feature,label,centers,batch_size的导数


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    ct = CenterLoss(10,2,size_average=True).to(device)
    y = torch.Tensor([0,0,2,1]).to(device)
    feat = torch.zeros(4,2).to(device).requires_grad_()
    print (list(ct.parameters()))
    print (ct.centers.grad)
    out = ct(y,feat)
    print(out.item())
    out.backward()
    print(ct.centers.grad)
    print(feat.grad)

if __name__ == '__main__':
    torch.manual_seed(999)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
