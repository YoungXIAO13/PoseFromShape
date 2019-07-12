import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import resnet


# ============================================================================ #
#                             Baseline network                                 #
# ============================================================================ #
class BaselineEstimator(nn.Module):
    """Pose estimator using image feature with shape feature

        Arguments:
        img_feature_dim: output feature dimension for image
        pretrained_resnet: use the ResNet pretrained on ImageNet if True

        Return:
        Three angle bin classification probability with a delta value regression for each bin
    """
    def __init__(self, img_feature_dim=1024, separate_branch=False,
                 azi_classes=24, ele_classes=12, inp_classes=24, pretrained_resnet=False):
        super(BaselineEstimator, self).__init__()

        # RGB image encoder
        self.img_encoder = resnet.resnet18(pretrained=pretrained_resnet, num_classes=img_feature_dim)

        self.compress = nn.Sequential(nn.Linear(img_feature_dim, 800), nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
        self.separate_branch = separate_branch

        # separate branch for classification and regression
        if separate_branch:
            self.compress_delta = nn.Sequential(nn.Linear(img_feature_dim, 800), nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                                nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                                nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))

        self.fc_cls_azi = nn.Linear(200, azi_classes)
        self.fc_cls_ele = nn.Linear(200, ele_classes)
        self.fc_cls_inp = nn.Linear(200, inp_classes)
        self.fc_reg_azi = nn.Linear(200, azi_classes)
        self.fc_reg_ele = nn.Linear(200, ele_classes)
        self.fc_reg_inp = nn.Linear(200, inp_classes)

    def forward(self, im):
        # pass the image through image encoder
        img_feature = self.img_encoder(im)

        # concatenate the features obtained from two encoders into one feature
        x = self.compress(img_feature)
        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)
        cls_inp = self.fc_cls_inp(x)

        # use the shared features if share branch
        x_delta = self.compress_delta(img_feature) if self.separate_branch else x
        reg_azi = self.fc_reg_azi(x_delta)
        reg_ele = self.fc_reg_ele(x_delta)
        reg_inp = self.fc_reg_inp(x_delta)
        return [cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp]


# ============================================================================ #
#                             Proposed network                                 #
# ============================================================================ #
class ShapeEncoderMV(nn.Module):
    """Shape Encoder using rendering images under multiple views

        Arguments:
        feature_dim: output feature dimension for each rendering image
        channels: 3 for normal rendering image, 4 for normal map with depth map, and 3*12 channels for concatenating
        pretrained_resnet: use the ResNet pretrained on ImageNet if True

        Return:
        A tensor of size NxC, where N is the batch size and C is the feature_dim
    """
    def __init__(self, feature_dim=256, channels=3, pretrained_resnet=False):
        super(ShapeEncoderMV, self).__init__()
        self.render_encoder = resnet.resnet18(input_channel=channels, num_classes=feature_dim, pretrained=pretrained_resnet)

    def forward(self, renders):
        # reshape render images from dimension N*K*C*H*W to (N*K)*C*H*W
        N, K, C, H, W = renders.size()
        renders = renders.view(N*K, C, H, W)

        # pass the encoder and reshape render features from dimension (N*K)*D1 to N*(K*D1)
        render_feature = self.render_encoder(renders)
        render_feature = render_feature.view(N, -1)
        return render_feature


class ShapeEncoderPC(nn.Module):
    """Shape Encoder using point cloud TO BE MODIFIED
    """
    def __init__(self, feature_dim=256, channels=3, pretrained_resnet=False):
        super(ShapeEncoderPC, self).__init__()
        self.pc_encoder = resnet.resnet18(input_channel=channels, num_classes=feature_dim, pretrained=pretrained_resnet)

    def forward(self, shapes):
        shape_feature = self.pc_encoder(shapes)
        return shape_feature


class PoseEstimator(nn.Module):
    """Pose estimator using image feature with shape feature

        Arguments:
        img_feature_dim: output feature dimension for image
        shape_feature_dim: output feature dimension for shape
        shape: shape representation in PointCloud or MultiView
        channels: channel number for multi-view encoder
        pretrained_resnet: use the ResNet pretrained on ImageNet if True

        Return:
        Three angle bin classification probability with a delta value regression for each bin
    """
    def __init__(self, render_number=12, img_feature_dim=1024, shape_feature_dim=256, channels=3, separate_branch=False,
                 azi_classes=24, ele_classes=12, inp_classes=24, pretrained_resnet=False, shape='PointCloud'):
        super(PoseEstimator, self).__init__()

        # 3D shape encoder
        if shape == 'PointCloud':
            self.shape_encoder = ShapeEncoderPC()
        else:
            self.shape_encoder = ShapeEncoderMV(feature_dim=shape_feature_dim, channels=channels, pretrained_resnet=pretrained_resnet)
        shape_feature_dim = shape_feature_dim * render_number if shape != 'PointCloud' else shape_feature_dim

        # RGB image encoder
        self.img_encoder = resnet.resnet18(pretrained=pretrained_resnet, num_classes=img_feature_dim)

        self.compress = nn.Sequential(nn.Linear(shape_feature_dim + img_feature_dim, 800),
                                      nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
        self.separate_branch = separate_branch

        # separate branch for classification and regression
        if separate_branch:
            self.compress_delta = nn.Sequential(nn.Linear(shape_feature_dim + img_feature_dim, 800),
                                                nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                                nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                                nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))	

        self.fc_cls_azi = nn.Linear(200, azi_classes)
        self.fc_cls_ele = nn.Linear(200, ele_classes)
        self.fc_cls_inp = nn.Linear(200, inp_classes)
        self.fc_reg_azi = nn.Linear(200, azi_classes)
        self.fc_reg_ele = nn.Linear(200, ele_classes)
        self.fc_reg_inp = nn.Linear(200, inp_classes)

    def forward(self, im, shape):
        # pass the image through image encoder
        img_feature = self.img_encoder(im)

        # pass the shape through shape encoder
        shape_feature = self.shape_encoder(shape)

        # concatenate the features obtained from two encoders into one feature
        global_feature = torch.cat((shape_feature, img_feature), 1)
        x = self.compress(global_feature)
        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)
        cls_inp = self.fc_cls_inp(x)

        # use the shared features if share branch
        x_delta = self.compress_delta(global_feature) if self.separate_branch else x
        reg_azi = self.fc_reg_azi(x_delta)
        reg_ele = self.fc_reg_ele(x_delta)
        reg_inp = self.fc_reg_inp(x_delta)
        return [cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp]


if __name__ == '__main__':
    print('test model')

    sim_im = Variable(torch.rand(4, 3, 224, 224))
    sim_renders = Variable(torch.rand(4, 12, 3, 224, 224))
    sim_im = sim_im.cuda()
    sim_renders = sim_renders.cuda()
    #model = PoseEstimator(shape='MultiView', separate_branch=False)
    model = BaselineEstimator(separate_branch=False, pretrained_resnet=False)
    model.cuda()
    #cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp = model(sim_im, sim_renders)
    cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp = model(sim_im)
    print(cls_azi.size(), cls_ele.size(), cls_inp.size(), reg_azi.size(), reg_ele.size(), reg_inp.size())
