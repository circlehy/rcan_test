from model import common

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = 6
        out_channels = 64
        depth = 7
        #bn = not gan_type == 'WGAN_GP'
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_du1 = nn.Sequential(
                nn.Conv2d(6, 1, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 3, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        m_features = [
            common.BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(common.BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        #x = torch.cat([x, y], 1)
        #x = conv_du1(x)  # back to 3 channel
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output

