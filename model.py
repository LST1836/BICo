import torch
from torch import nn
from transformers import AutoModel
from layer import HieTree, TextFacetMatching, TextFacetMatchingMH, Gate, MLP


class RelevanceNet(nn.Module):
    def __init__(self, arg, tree_structure, ids_concept, mask_concept):
        super(RelevanceNet, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(arg.plm_path)

        self.dim = self.text_encoder.config.hidden_size

        if arg.concept_tree:
            self.hie_tree = HieTree(self.dim, arg.num_iter_hie_tree)

        if arg.mh_matching:
            self.text_facet_match_1 = TextFacetMatchingMH(self.dim, arg.num_head_matching)
            if arg.concept_tree:
                self.text_facet_match_2 = TextFacetMatchingMH(self.dim, arg.num_head_matching)
        else:
            self.text_facet_match_1 = TextFacetMatching(self.dim)
            if arg.concept_tree:
                self.text_facet_match_2 = TextFacetMatching(self.dim)

        if arg.concept_tree:
            self.gates = nn.ModuleList([Gate(self.dim) for i in range(12)])

        self.mlps = nn.ModuleList([MLP(self.dim, arg.cls_num, arg.dropout_mlp, arg.act_mlp) for i in range(12)])

        self.cos_sim = nn.CosineSimilarity(dim=2)  # CL
        self.softmax = nn.Softmax(dim=1)  # CL

        self.ids_concept = ids_concept
        self.mask_concept = mask_concept

        self.arg = arg
        self.tree_structure = tree_structure

    def forward(self, ids_text, mask_text, label=None):
        text_embed = self.text_encoder(input_ids=ids_text, attention_mask=mask_text).last_hidden_state  # [b, s, d]

        concept_rep = self.text_encoder(input_ids=self.ids_concept.cuda(),
                                           attention_mask=self.mask_concept.cuda()).last_hidden_state[:, 0, :]  # [48, d]
        root_rep, domain_rep = self.root_domain_rep_from_facet(concept_rep[:12])
        concept_rep = torch.cat((root_rep, domain_rep, concept_rep), dim=0)  # [54, d]

        if self.arg.concept_tree:
            updated_concept_rep = self.hie_tree(concept_rep, self.tree_structure)  # [54, d]

        facet_aware_text_rep_1 = self.text_facet_match_1(text_embed, concept_rep[6:18], mask_text)  # [b, 12, d]
        facet_aware_text_rep_1 = facet_aware_text_rep_1.transpose(0, 1)  # [12, b, d]
        if self.arg.concept_tree:
            facet_aware_text_rep_2 = self.text_facet_match_2(text_embed, updated_concept_rep[6:18], mask_text)
            facet_aware_text_rep_2 = facet_aware_text_rep_2.transpose(0, 1)

        cl_loss_list, logits_list = [], []
        for i in range(12):
            if self.arg.concept_tree:
                text_rep = self.gates[i](facet_aware_text_rep_1[i], facet_aware_text_rep_2[i])  # [b, d]
            else:
                text_rep = facet_aware_text_rep_1[i]  # [b, d]

            if self.training:
                cl_loss = self.cl(text_rep, label.transpose(0, 1)[i], self.arg.t1[i])
                cl_loss_list.append(cl_loss.unsqueeze(0))

            logits = self.mlps[i](text_rep)  # [b, 4]
            logits_list.append(logits.unsqueeze(0))  # [1, b, 4]

        logits = torch.cat(logits_list, dim=0)  # [12, b, 4]
        return logits.permute(1, 2, 0), cl_loss_list  # [b, 4, 12]

    def root_domain_rep_from_facet(self, facet_rep):
        domain_rep = torch.zeros((5, self.dim)).cuda()
        s = 0
        for i, n in enumerate(self.tree_structure[2]):
            domain_rep[i] = torch.mean(facet_rep[s:s+n], dim=0)
            s = s + n

        root_rep = torch.mean(domain_rep, dim=0, keepdim=True)  # [1, d]

        return root_rep, domain_rep

    def cl(self, text_rep, label, t1):  # Contrastive Learning
        bs = text_rep.shape[0]

        if torch.sum(label != 0) == 0:
            return torch.tensor(0).cuda()

        sim = self.cos_sim(text_rep.unsqueeze(1).expand(-1, bs, -1),
                           text_rep.unsqueeze(0).expand(bs, -1, -1))
        sim = sim / t1
        mask = -1e30 * torch.eye(bs).cuda()
        sim = sim + mask
        sim = self.softmax(sim)

        related_cl_losses = []
        for i in range(bs):
            pos_mask = label == label[i]
            pos_mask[i] = False
            pos_num = torch.sum(pos_mask)
            if pos_num == 0:
                continue
            related_cl_losses.append(torch.log(torch.sum(sim[i][pos_mask])))

        if len(related_cl_losses) == 0:
            return torch.tensor(0).cuda()
        else:
            related_cl_loss = sum(related_cl_losses) / len(related_cl_losses)
            return -related_cl_loss


class IdeologyNet(nn.Module):
    def __init__(self, arg, tree_structure, ids_concept, mask_concept):
        super(IdeologyNet, self).__init__()

        self.text_encoder = AutoModel.from_pretrained(arg.plm_path)

        self.dim = self.text_encoder.config.hidden_size

        if arg.concept_tree:
            self.hie_tree = HieTree(self.dim, arg.num_iter_hie_tree)

        self.mlps = nn.ModuleList([MLP(self.dim, arg.cls_num, arg.dropout_mlp, arg.act_mlp) for i in range(12)])
        self.ideo_rep_linear = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=False) for i in range(12)])

        self.cos_sim = nn.CosineSimilarity(dim=2)  # CL
        self.softmax = nn.Softmax(dim=1)  # CL

        self.ids_concept = ids_concept
        self.mask_concept = mask_concept

        self.arg = arg
        self.tree_structure = tree_structure

    def forward(self, ids_text, mask_text, facet_idx, label=None):
        text_rep = self.text_encoder(input_ids=ids_text, attention_mask=mask_text).last_hidden_state[:, 0, :]  # [b, d]

        concept_rep = self.text_encoder(input_ids=self.ids_concept.cuda(),
                                           attention_mask=self.mask_concept.cuda()).last_hidden_state[:, 0, :]  # [48, d]
        root_rep, domain_rep = self.root_domain_rep_from_facet(concept_rep[:12])
        concept_rep = torch.cat((root_rep, domain_rep, concept_rep), dim=0)  # [54, d]

        if self.arg.concept_tree:
            updated_concept_rep = self.hie_tree(concept_rep, self.tree_structure)  # [54, d]
            ideo_rep = updated_concept_rep[-36:]  # [36, d]
        else:
            ideo_rep = concept_rep[-36:]

        cl_loss_list, logits_list = [], []
        for i in range(12):
            facet_mask = facet_idx == i
            facet_num = torch.sum(facet_mask)
            if facet_num == 0:
                cl_loss_list.append(torch.tensor([0]).cuda())
                continue

            text_rep_i = text_rep[facet_mask]

            if self.training:
                ideo_rep_i = ideo_rep[3 * i:3 * i + 3]
                ideo_rep_i = self.ideo_rep_linear[i](ideo_rep_i)
                cl_loss = self.concept_guided_cl(text_rep_i, label[facet_mask], ideo_rep_i, self.arg.t2[i])
                cl_loss_list.append(cl_loss.unsqueeze(0))

            logits = self.mlps[i](text_rep_i)  # [n, 4]
            logits_list.append(logits)

        return logits_list, cl_loss_list

    def root_domain_rep_from_facet(self, facet_rep):
        domain_rep = torch.zeros((5, self.dim)).cuda()
        s = 0
        for i, n in enumerate(self.tree_structure[2]):
            domain_rep[i] = torch.mean(facet_rep[s:s+n], dim=0)
            s = s + n

        root_rep = torch.mean(domain_rep, dim=0, keepdim=True)  # [1, d]

        return root_rep, domain_rep

    def concept_guided_cl(self, text_rep, label, ideo_rep, t):
        bs = text_rep.shape[0]
        text_rep = torch.cat((text_rep, ideo_rep), dim=0)  # [b+3, d]
        label = torch.cat((label, torch.LongTensor([0, 1, 2]).cuda()), dim=0)  # [b+3]

        sim = self.cos_sim(text_rep.unsqueeze(1).expand(-1, bs + 3, -1),
                           text_rep.unsqueeze(0).expand(bs + 3, -1, -1))  # [b+3, b+3]
        sim = sim / t
        mask = -1e30 * torch.eye(bs + 3).cuda()
        sim = sim + mask
        sim = self.softmax(sim)

        cl_losses = []
        for i in range(bs + 3):
        # for i in [bs, bs+1, bs+2]:
            pos_mask = label == label[i]
            pos_mask[i] = False
            pos_num = torch.sum(pos_mask)
            if pos_num == 0:
                continue
            cl_losses.append(torch.log(torch.sum(sim[i][pos_mask])))
        if len(cl_losses) == 0:
            return torch.tensor(0).cuda()
        else:
            return -sum(cl_losses) / len(cl_losses)

