import torch.nn as nn
import torch.nn.functional as F
import torch
from .tag_net import TagNet
from .vidual_net import VidualNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def similarity_x_c(x, W_c, gamma=.2):
#     base = torch.mm(x, W_c.transpose(1, 0))
#     exp_base = (1/gamma*base).exponential_()
#     sum_k = exp_base.sum(dim=1)

#     similarity = exp_base*base/sum_k.unsqueeze(dim=1)
#     similarity = similarity.sum(dim=1)
#     return similarity


def similarity_x(x, W, n_categories, n_multi_center, gamma=.2):
    base = F.linear(x, W).view(-1, n_categories, n_multi_center)
    similarity = F.softmax(1/gamma*base, dim=1).mul(base).sum(dim=2)
    return similarity


def soft_triple(x, W, cat, C, K, gamma=.1, lambda_=.1, margin=.01):
    similaritys = similarity_x(x, W, C, K, gamma=gamma)
    one_hot = torch.zeros(similaritys.size(), device=device)
    one_hot.scatter_(1, cat.view(-1,1).long(), 1)
    similaritys = similaritys - margin*one_hot
    similaritys = lambda_* similaritys
    # class loss
    return F.cross_entropy(similaritys, cat)


def regularizer(W, C, K, tau=.2):
    def reg(w_sub):
        sub_norm = 1.0 - torch.matmul(w_sub, w_sub.transpose(1, 0))
        sub_norm = torch.clamp(sub_norm, min=1e-10)
        return torch.sqrt(2*sub_norm.triu(diagonal=1)).sum()

    L_2_1 = None
    for c in range(C):
        r_sum = reg(W[c*K:(c+1)*K])
        if L_2_1 is None:
            L_2_1 = r_sum
        else:
            L_2_1 += r_sum

    out = tau * L_2_1
    out /= C*K*(K-1)
    return out


def dist(x, W, C, K, target_category, tau=1e-2):
    st = soft_triple(x, W, target_category, C, K).mean()
    reg = regularizer(W, C, K)
    # out = st + tau*reg
    out = st + reg
    return out


class TransNFCM(nn.Module):
    def __init__(self, in_ch, out_ch, n_category,
                 embedding_dim=128, K=2):
        super().__init__()
        self.W = torch.nn.Parameter(data=torch.Tensor(n_category*K, embedding_dim*2))
                                    #requires_grad=True)
        nn.init.normal_(self.W, 0.0, 1./embedding_dim*2)
        self.vidual_net = VidualNet(in_ch, out_ch)
        # TODO: more efficientry
        self.tag_net = TagNet(n_category, embedding_dim)
        self.C = n_category
        # number of center per class
        self.K = K

    def calculate_VT_encoded_vec(self, v_vec, t_vec):
        v_vec = F.normalize(v_vec, p=2, dim=1)
        t_vec = F.normalize(t_vec, p=2, dim=1)
        vt_vec = torch.cat([v_vec, v_vec], dim=1)
        return vt_vec

    def calculate_distance(self, x, cat):
        x_vidual_vec = self.vidual_net(x)
        x_tag_vec = self.tag_net(cat)
        x_vt = self.calculate_VT_encoded_vec(x_vidual_vec, x_tag_vec)
        weight = F.normalize(self.W, p=2, dim=1)
        return dist(x_vt, weight, self.C, self.K, cat)

    def predict(self, x=None, category=None):
        if x is None and category is not None:
            return F.normalize(self.tag_net(category), p=2, dim=1)
        elif x is not None and category is None:
            return F.normalize(self.vidual_net(x), p=2, dim=1)
        elif x is not None and category is not None:
            vidual_vec = self.vidual_net(x)
            tag_vec = self.tag_net(category)
            vt = self.calculate_VT_encoded_vec(vidual_vec, tag_vec)
            return F.normalize(vt, p=2, dim=1)
        else:
            assert False, "image or category, either one is required"

    def forward(self, x, cat):
        return self.calculate_distance(x, cat)
