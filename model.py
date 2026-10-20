from torch.nn import Module

from candidate_graph import AssociationGraphMatcher


class CongMatch(Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = AssociationGraphMatcher(
            hidden_dim=args.candidate_hidden_dim,
            layer_count=args.candidate_num_layers,
            head_count=args.association_heads,
            dropout=args.candidate_dropout,
            camh_gamma=args.camh_gamma_c,
        )
        self.latest_constraints = None

    def candidate_logits(self, candidate_batch):
        _, logits = self.matcher(candidate_batch)
        self.latest_constraints = self.matcher.latest_constraint_losses
        return logits

    def candidate_constraint_losses(self):
        return self.latest_constraints
