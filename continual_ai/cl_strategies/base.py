import torch


class Container(object):
    def __init__(self):

        self.encoder = None
        self.solver = None
        self.other_models = torch.nn.ModuleDict()
        self.optimizer = None

        self.current_loss = None

        self.current_task = None
        self.current_batch = None
        self.current_epoch = None
        self.num_tasks = None

        self.others_parameters = dict()


class NaiveMethod(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(NaiveMethod, self).__init__()

    def on_epoch_starts(self, container: Container, *args, **kwargs):
        pass

    def on_epoch_ends(self, container: Container,  *args, **kwargs):
        pass

    def on_task_starts(self, container: Container, *args, **kwargs):
        pass

    def on_task_ends(self,container: Container,  *args, **kwargs):
        pass

    def on_batch_starts(self, container: Container, *args, **kwargs):
        pass

    def after_optimization_step(self, container: Container,  *args, **kwargs):
        pass

    def after_back_propagation(self, container: Container, *args, **kwargs):
        pass

    def before_gradient_calculation(self, container: Container,  *args, **kwargs):
        pass

