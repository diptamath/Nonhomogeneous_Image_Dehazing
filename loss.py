import torch
from torch import nn
from torchvision.models.vgg import vgg16
from skimage.measure import compare_psnr, compare_ssim
import pytorch_ssim
from torch.autograd import Variable

class CustomLoss_function(nn.Module):
    def __init__(self):
        super(CustomLoss_function, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):

        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        recons_loss = 0.6*self.mae_loss(out_images, target_images) + 0.4*self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)

        loss = recons_loss + 0.006*perception_loss + 2e-8*tv_loss

        return loss, recons_loss, perception_loss, tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = CustomLoss_function()
    print(g_loss)


