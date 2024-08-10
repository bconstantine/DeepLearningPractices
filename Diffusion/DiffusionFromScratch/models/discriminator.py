import torch

from .settings import DiscriminatorConfig

class PatchGANDiscriminator(torch.nn.Module):
    """Output of the Discriminator in Patch rather than a scalar as used in PatchGAN:
    Which have advantage such as:
    1. Local focus
    2. More flexible image sizes"""
    def __init__(self, discriminator_config: DiscriminatorConfig, im_channels:int):
        """Model implementation follows:
        https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/models/discriminator.py"""

        super().__init__()
        self.im_channels = im_channels
        channels_dim = [self.im_channels] + discriminator_config.conv_channels_list + [1]
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(channels_dim[i], channels_dim[i + 1],
                          kernel_size=discriminator_config.kernel_size_list[i],
                          stride=discriminator_config.stride_list[i],
                          padding=discriminator_config.padding_list[i],
                          bias=False if i !=0 else True),
                #place batch norm on all layers except the first and last,
                #note that the batch distribution (all real, all fake, separated or randomized in same batch) may affect this result
                #that is why, the first layer is not batch normed, let the model learn the distribution
                #the last layer is also not batch normed, to make the output more contrast between fake and real (easier to learn)
                #Shoutout to this link for nice insight: https://ovgu-ailab.github.io/blog/methods/2022/07/07/batchnorm-gans.html
                torch.nn.BatchNorm2d(channels_dim[i + 1]) if i != len(channels_dim) - 2 and i != 0 else torch.nn.Identity(),
                #place leaku relu on all layers except the last
                torch.nn.LeakyReLU(0.2) if i != len(channels_dim) - 2 else torch.nn.Identity()
            )
            for i in range(len(channels_dim) - 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
