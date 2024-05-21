import torch
from torch import nn
import math


class TextFacetMatching(nn.Module):
    def __init__(self, dim):
        super(TextFacetMatching, self).__init__()

        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        self.o_linear = nn.Linear(dim, dim, bias=False)

        # self.cos_sim = nn.CosineSimilarity(dim=3)

        self.softmax = nn.Softmax(dim=-1)

        self.dim = dim

    def forward(self, text_embed, facet_embed, att_mask):
        # text_embed: [b, s, e]  facet_embed: [12, e]
        q = self.q_linear(facet_embed)  # [12, e]
        k = self.k_linear(text_embed)  # [b, s, e]
        v = self.v_linear(text_embed)  # [b, s, e]

        # [b, 1, s, e] * [12, e, 1] = [b, 12, s, 1]
        att_score = torch.matmul(k.unsqueeze(dim=1), q.unsqueeze(dim=-1)) / math.sqrt(self.dim)
        att_score = att_score.squeeze(dim=-1)  # [b, 12, s]
        # att_score = self.cos_sim(q.unsqueeze(1).unsqueeze(0).expand(k.shape[0], -1, k.shape[1], -1),
        #                          k.unsqueeze(1).expand(-1, 12, -1, -1))  # [b, 12, s]

        att_score = torch.masked_fill(att_score, att_mask.unsqueeze(1) == 0, -1e30)

        att_score = self.softmax(att_score)  # [b, 12, s]

        out = torch.matmul(att_score, v)  # [b, 12, e]
        out = self.o_linear(out)

        return out


class TextFacetMatchingMH(nn.Module):
    def __init__(self, hidden_dim, num_head):
        super(TextFacetMatchingMH, self).__init__()

        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

    def forward(self, text_embed, facet_embed, att_mask):
        # text_embed: [b, s, e]  facet_embed: [12, e]
        text_shape = text_embed.shape
        facet_shape = facet_embed.shape
        head_q = self.q_linear(facet_embed).view(facet_shape[0], self.num_head, self.head_dim)  # [12, num_head, head_dim]
        head_k = self.k_linear(text_embed).view(text_shape[0], text_shape[1], self.num_head, self.head_dim)  # [b, s, num_head, head_dim]
        head_v = self.v_linear(text_embed).view(text_shape[0], text_shape[1], self.num_head, self.head_dim)  # [b, s, num_head, head_dim]

        # [b, num_head, s, head_dim] * [num_head, head_dim, 12] = [b, num_head, s, 12]
        att_score = torch.matmul(head_k.transpose(1, 2), head_q.permute(1, 2, 0)) / math.sqrt(self.head_dim)
        att_score = att_score.transpose(2, 3)  # [b, num_head, 12, s]
        att_score = torch.masked_fill(att_score, att_mask.unsqueeze(1).unsqueeze(1) == 0, -1e30)

        att_score = self.softmax(att_score)

        out = torch.matmul(att_score, head_v.transpose(1, 2))  # [b, num_head, 12, head_dim]
        out = out.transpose(1, 2).contiguous().view(text_shape[0], -1, self.num_head*self.head_dim)  # [b, 12, num_head*head_dim]
        out = self.o_linear(out)

        return out


class HieTree(nn.Module):
    def __init__(self, hidden_dim, num_iter):
        super(HieTree, self).__init__()

        pi = 3.14159265358979323846

        self.edge_embed = nn.Parameter(torch.zeros(3, hidden_dim // 2))
        nn.init.uniform_(self.edge_embed, a=-pi, b=pi)

        self.metapath_aggr_layers = nn.ModuleList([MetapathAggr(hidden_dim) for i in range(num_iter)])
        self.hie_gat_layers = nn.ModuleList([HieGATLayer(hidden_dim) for i in range(num_iter)])

        self.num_iter = num_iter

    def forward(self, concept_embed, tree_structure):
        edge_real = torch.cos(self.edge_embed)
        edge_imag = torch.sin(self.edge_embed)
        for i in range(self.num_iter):
            concept_embed = self.hie_gat_layers[i](concept_embed, tree_structure)
            concept_embed = self.metapath_aggr_layers[i](concept_embed, edge_real, edge_imag, tree_structure)

        return concept_embed


class MetapathAggr(nn.Module):
    def __init__(self, hidden_dim):
        super(MetapathAggr, self).__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, concept_embed, edge_real, edge_imag, tree_structure):
        concept_embed = self.linear(concept_embed)
        concept_real, concept_imag = torch.chunk(concept_embed, 2, dim=1)

        root_real, root_imag = concept_real[0].unsqueeze(0), concept_imag[0].unsqueeze(0)  # [1, d]
        domain_real, domain_imag = concept_real[1:6], concept_imag[1:6]  # [5, d]
        edge1_real, edge1_imag = edge_real[0].unsqueeze(0), edge_imag[0].unsqueeze(0)  # [1, d]
        domain_real = domain_real + (root_real * edge1_real - root_imag * edge1_imag)
        domain_imag = domain_imag + (root_real * edge1_imag + root_imag * edge1_real)

        facet_real, facet_imag = concept_real[6:18], concept_imag[6:18]  # [12, d]
        facet_real_new, facet_imag_new = facet_real.clone(), facet_imag.clone()
        edge2_real, edge2_imag = edge_real[1], edge_imag[1]  # [d]
        structure = tree_structure[2]  # [2, 2, 3, 2, 3]
        s = 0
        for i, n in enumerate(structure):
            facet_real_new[s:s+n] = facet_real[s:s+n] + \
                                    (domain_real[i] * edge2_real - domain_imag[i] * edge2_imag).unsqueeze(0)
            facet_imag_new[s:s+n] = facet_imag[s:s+n] + \
                                    (domain_real[i] * edge2_imag + domain_imag[i] * edge2_real).unsqueeze(0)
            s = s + n

        ideo_real, ideo_imag = concept_real[18:54], concept_imag[18:54]  # [36, d]
        ideo_real_new, ideo_imag_new = ideo_real.clone(), ideo_imag.clone()
        edge3_real, edge3_imag = edge_real[2], edge_imag[2]  # [d]
        structure = tree_structure[3]  # [3, 3, 3, 3, ..., 3]
        s = 0
        for i, n in enumerate(structure):
            ideo_real_new[s:s+n] = ideo_real[s:s+n] + \
                                   (facet_real_new[i] * edge3_real - facet_imag_new[i] * edge3_imag).unsqueeze(0)
            ideo_imag_new[s:s+n] = ideo_imag[s:s+n] + \
                                   (facet_real_new[i] * edge3_imag + facet_imag_new[i] * edge3_real).unsqueeze(0)
            s = s + n

        domain_embed_new = torch.cat((domain_real, domain_imag), dim=1) / 2
        facet_embed_new = torch.cat((facet_real_new, facet_imag_new), dim=1) / 3
        ideo_embed_new = torch.cat((ideo_real_new, ideo_imag_new), dim=1) / 4
        concept_embed_new = torch.cat((concept_embed[0].unsqueeze(0),
                                       domain_embed_new, facet_embed_new, ideo_embed_new), dim=0)

        return concept_embed_new


class HieGATLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(HieGATLayer, self).__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_if = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.att_fd = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.att_dr = nn.Linear(2 * hidden_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)

        self.hidden_dim = hidden_dim

    def forward(self, concept_embed, tree_structure):
        concept_embed = self.linear(concept_embed)

        facet_embed = concept_embed[6:18]  # [12, d]
        facet_embed_new = facet_embed.clone()
        ideo_embed = concept_embed[18:54]  # [36, d]
        structure = tree_structure[3]
        s = 0
        for i, n in enumerate(structure):
            facet_embed_new[i] = self.attn_aggr(facet_embed[i], ideo_embed[s:s+n], 'if')
            s = s + n

        domain_embed = concept_embed[1:6]
        domain_embed_new = domain_embed.clone()
        structure = tree_structure[2]
        s = 0
        for i, n in enumerate(structure):
            domain_embed_new[i] = self.attn_aggr(domain_embed[i], facet_embed_new[s:s+n], 'fd')
            s = s + n

        root_embed_new = self.attn_aggr(concept_embed[0], domain_embed_new, 'dr')

        concept_embed_new = torch.cat((root_embed_new.unsqueeze(0), domain_embed_new, facet_embed_new, ideo_embed),
                                      dim=0)
        return concept_embed_new

    def attn_aggr(self, center_node, child_node, hierarchy='if'):
        child_node = torch.cat((center_node.unsqueeze(0), child_node), dim=0)
        att_pre = torch.cat((center_node.unsqueeze(0).expand(child_node.shape[0], -1), child_node), dim=1)  # [n, 2d]

        if hierarchy == 'if':
            att_score = self.att_if(att_pre)
        elif hierarchy == 'fd':
            att_score = self.att_fd(att_pre)
        elif hierarchy == 'dr':
            att_score = self.att_dr(att_pre)
        else:
            raise KeyError

        att_score = self.leaky_relu(att_score)  # [n, 1]
        att_score = self.softmax(att_score)

        center_node_new = torch.matmul(att_score.transpose(0, 1), child_node)  # [1, d]

        return center_node_new.squeeze(0)


class Gate(nn.Module):
    def __init__(self, hidden_dim):
        super(Gate, self).__init__()

        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, rep1, rep2):
        rep = torch.cat((rep1, rep2), dim=-1)  # [b, 2d]

        a1 = torch.sigmoid(self.linear1(rep))
        a2 = torch.sigmoid(self.linear2(rep))

        return a1 * rep1 + a2 * rep2


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, activation):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_dim, 512)
        self.linear2 = nn.Linear(512, out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

