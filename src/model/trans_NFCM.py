import torch.nn as nn
import torch.nn.functional as F
import torch
from .tag_net import TagNet
from .vidual_net import VidualNet


def similarity_x_c(x, W_c, gamma=.2):
    base = torch.mm(x, W_c.transpose(1, 0))
    exp_base = (1/gamma*base).exponential_()
    sum_k = exp_base.sum(dim=1)

    similarity = exp_base*base/sum_k.unsqueeze(dim=1)
    similarity = similarity.sum(dim=1)
    return similarity


def soft_triple(x, W, cat, K, gamma=.1, lambda_=.1, margin=.01):
    W_len, _ = W.shape
    n_cat = int(W_len//K)

    def expt_y_cat(x_cat):
        out = None
        for i_cat in range(n_cat):
            if i_cat!=x_cat:
                w_c = W[i_cat*K:(i_cat+1)*K]
                w_c /= w_c.sum(dim=1).unsqueeze(1)
                o = lambda_*(similarity_x_c(x, w_c, gamma=gamma))
                if out is None:
                    out = o
                else:
                    out += o
        return out

    def base_cat(w_c):
        return lambda_*(similarity_x_c(x, w_c, gamma=gamma) - margin)

    base_y_cat = torch.cat([base_cat(W[c*K:(c+1)*K]) for c in cat], 0)
    base_y_cat = base_y_cat.exponential_()
    expt_y_cat = torch.cat([expt_y_cat(c) for c in cat])
    return -1 * (1 + base_y_cat/(base_y_cat + expt_y_cat)).log()


def regularizer(W, C, K, tau=.2):
    def each(w_c):
        r_sum = None
        for k in range(K-1):
            r = 2 - 2 * torch.dot(w_c[k]/w_c[k].sum(), w_c[k+1]/w_c[k+1].sum())
            r = r.sqrt()
            if r_sum is None:
                r_sum = r
            else:
                r_sum += r
        return r_sum

    L_2_1 = None
    for c in range(C):
        r_sum = each(W[c*K:(c+1)*K])
        print("r_sum:")
        print(r_sum)
        if L_2_1 is None:
            L_2_1 = r_sum
        else:
            L_2_1 += r_sum
        print("L_2_1:")
        print(L_2_1)

    out = tau * L_2_1
    out /= C*K*(K-1)
    print("out:")
    print(out)
    return out


def dist(x, W, C, K, target_category):
    n_batch, _ = x.shape
    st = soft_triple(x, W, target_category, K).mean()
    print("st:")
    print(st)
    reg = regularizer(W, C, K)
    print("reg:")
    print(reg)
    out = st + reg
    return out


class TransNFCM(nn.Module):
    def __init__(self, in_ch, out_ch, n_category,
                 embedding_dim=128, K=2):
        super().__init__()
        self.W = torch.nn.Parameter(data=torch.Tensor(n_category*K, embedding_dim*2),
                                    requires_grad=True)
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
        return dist(x_vt, self.W, self.C, self.K, cat)

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
