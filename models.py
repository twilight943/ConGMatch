import torch
from torch.nn import Linear, Module

from candidate_invariance import AssociationGraphMatcher


class NC(Module):
    def __init__(self, args):
        super().__init__()
        self.candidate_pi_mode = "association"
        self.candidate_model = AssociationGraphMatcher(
            hidden_dim=args.candidate_hidden_dim,
            num_layers=args.candidate_num_layers,
            dropout=args.candidate_dropout,
            transformer_heads=args.association_heads,
            pool=args.association_pool,
        )
        self.predictor = Linear(1, 2)

    def candidate_logits(self, candidate_batch):
        _, logits = self.candidate_model(candidate_batch)
        return logits

    def candidate_score(self, candidate_batch):
        return torch.sigmoid(self.candidate_logits(candidate_batch))
