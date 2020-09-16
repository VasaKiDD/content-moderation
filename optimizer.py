import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


class Metric(Module):
    def forward(self, *input):
        raise NotImplementedError


class ClassificationAccuracy(Metric):
    def __init__(self):
        super(ClassificationAccuracy, self).__init__()

    def forward(self, y_pred, y_true):
        rewards = 0.0
        for i in range(y_pred.size(0)):
            _sort, ind = torch.sort(y_pred[i], descending=True)
            if torch.equal(y_true[i], ind[0]):
                rewards += 1.0
        return rewards


class SpecialClassificationAccuracy(Metric):
    def __init__(self):
        super(SpecialClassificationAccuracy, self).__init__()

    def forward(self, y_pred, y_true):
        rewards = 0.0
        non_null = 0.0
        for i in range(y_pred.size(0)):
            _, ind = torch.sort(y_pred[i], descending=True)
            sort, ind2 = torch.sort(y_true[i], descending=True)
            if sort[0] > 0.0:
                if torch.equal(ind2[0], ind[0]):
                    rewards += 1.0
                non_null += 1.0
        return rewards, non_null


class ParamOptimizer(Module):
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGDCosineAnnealed(ParamOptimizer):
    def __init__(
        self,
        model,
        batch_size,
        train_size,
        momentum=0.9,
        lmax=0.05,
        lmin=0.001,
        T0=10,
        l2=1e-4,
        gradient_clip=5,
    ):
        super(SGDCosineAnnealed, self).__init__()
        self.model = model
        self.optimizer = SGD(
            model.parameters(), lr=lmax, momentum=momentum, weight_decay=l2, nesterov=True
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T0, eta_min=lmin)
        # self.scheduler.step()
        self.Tn = T0
        self.steps_in_epoch = train_size // batch_size
        self.cpt_steps = 0
        self.gradient_clip = gradient_clip

    def step(self):
        clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.cpt_steps += 1
        if self.cpt_steps == self.steps_in_epoch:
            self.cpt_steps = 0
            self.scheduler.step()
            if self.scheduler.last_epoch == self.Tn:
                self.Tn = self.Tn * 2
                self.scheduler.T_max = self.Tn
                self.scheduler.last_epoch = 0

    def zero_grad(self):
        self.optimizer.zero_grad()
