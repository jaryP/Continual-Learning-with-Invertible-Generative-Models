import torch.nn.functional as F


class KnowledgeDistillationLoss:
    def __init__(self, temperature):
        self.t = temperature

    def __call__(self, prediction, target):
        soft_log_probs = F.log_softmax(prediction / self.t, dim=-1)
        soft_targets = F.softmax(target / self.t, dim=-1)

        distillation_loss = F.kl_div(soft_log_probs, soft_targets, reduction='batchmean')

        distillation_loss = distillation_loss * self.t ** 2

        return distillation_loss
