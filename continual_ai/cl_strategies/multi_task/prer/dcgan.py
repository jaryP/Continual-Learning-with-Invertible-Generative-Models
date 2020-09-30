import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)


class DCGAN(torch.nn.Module):
    def __init__(self, emb_size, hidden_dimension, depth, cond_dim=0, z_dim=None):
        super().__init__()

        if z_dim is None:
            z_dim = emb_size

        self.generator = [torch.nn.Linear(z_dim + cond_dim, hidden_dimension), torch.nn.ReLU()]

        for i in range(depth):
            self.generator.append(torch.nn.Linear(hidden_dimension, hidden_dimension))
            self.generator.append(torch.nn.ReLU())

        self.generator.append(torch.nn.Linear(hidden_dimension, emb_size))
        self.generator = torch.nn.Sequential(*self.generator)

        self.discriminator = torch.nn.Sequential(*[
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size // 2, 1),
            torch.nn.Sigmoid()
        ])

        self.discriminator.apply(weights_init)

    def G(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=1)

        emb_hat = self.generator(x)

        return emb_hat

    def D(self, x, y=None):
        emb_class = self.discriminator(x)
        return emb_class

    def forward(self, x, y=None):
        emb_hat = self.G(x, y)
        emb_class = self.D(emb_hat)

        return emb_hat, emb_class

    def backward(self, x, y=None):
        return self.forward(x, y)
