import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import resnet


class BaselineNet(nn.Module):
    def __init__(self, real_feature_dim=1024, azi_classes=24, ele_classes=12, inp_classes=24, all_angles=True,
                 features=64, input_dim=224, pretrained_resnet=False, res_pooling=True):
        super(BaselineNet, self).__init__()
        self.all_angles = all_angles
        last_layer = -1 if res_pooling else -2
        last_layer_dim = 1 if res_pooling else int(input_dim / 32) ** 2

        real_encoder = resnet.resnet18(pretrained=pretrained_resnet, input_channel=3, num_classes=real_feature_dim,
                                features=features)
        self.real_encoder = nn.Sequential(*list(real_encoder.children())[:last_layer])
        self.fc = nn.Sequential(nn.Linear(last_layer_dim*features*8, real_feature_dim),
                                nn.BatchNorm1d(real_feature_dim), nn.ReLU(inplace=True),
                                nn.Linear(real_feature_dim, 800), nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(200, azi_classes)
        self.fc4 = nn.Linear(200, ele_classes)
        self.fc5 = nn.Linear(200, inp_classes)

    def forward(self, im):
        real_feature = self.real_encoder(im)
        real_feature = real_feature.view(real_feature.size(0), -1)
        x = self.fc(real_feature)
        out_azi = self.fc3(x)
        out_ele = self.fc4(x)
        if self.all_angles:
            out_inp = self.fc5(x)
            return [out_azi, out_ele, out_inp]
        else:
            return [out_azi, out_ele]


class MultiViewNet(nn.Module):
    def __init__(self, render_number=24, render_feature_dim=256, real_feature_dim=1024,
                 azi_classes=24, ele_classes=12, inp_classes=24, all_angles=True, channels=3,
                 features=16, input_dim=224, pretrained_resnet=False, view_pooling=False, res_pooling=False, res=18):
        super(MultiViewNet, self).__init__()
        self.render_feature_dim = render_feature_dim
        self.render_number = render_number
        self.all_angles = all_angles
        self.view_pooling = view_pooling

        ResNet = resnet.resnet18 if res == 18 else (resnet.resnet34 if res == 34 else resnet.resnet50)
        expansion = 4 if res == 50 else 1
        f_render = features
        last_layer = -1 if res_pooling else -2
        last_layer_dim = 1 if res_pooling else int(input_dim / 32) ** 2
        render_encoder = ResNet(input_channel=channels, num_classes=render_feature_dim, features=f_render)
        self.render_encoder = nn.Sequential(*list(render_encoder.children())[:last_layer])
        self.fc1 = nn.Sequential(nn.Linear(last_layer_dim*f_render*8*expansion, render_feature_dim),
                                 nn.BatchNorm1d(render_feature_dim), nn.ReLU(inplace=True))

        real_encoder = ResNet(pretrained=pretrained_resnet, input_channel=3, num_classes=real_feature_dim, features=features)
        self.real_encoder = nn.Sequential(*list(real_encoder.children())[:last_layer])
        self.fc2 = nn.Sequential(nn.Linear(last_layer_dim*features*8*expansion, real_feature_dim),
                                 nn.BatchNorm1d(real_feature_dim), nn.ReLU(inplace=True))

        self.pooling = nn.AdaptiveMaxPool1d(1)
        fc_combine = nn.Linear(render_feature_dim + real_feature_dim, 800) if view_pooling else nn.Linear(render_number * render_feature_dim + real_feature_dim, 800)
        self.compress = nn.Sequential(fc_combine, nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))

        #self.fc3 = nn.Linear(200, azi_classes)
        #self.fc4 = nn.Linear(200, ele_classes)

        self.fc4 = nn.Linear(200, azi_classes)
        self.fc3 = nn.Linear(200, ele_classes)

        self.fc5 = nn.Linear(200, inp_classes)

    def forward(self, im, renders):
        # reshape render images from dimension N*K*C*H*W to (N*K)*C*H*W
        shape = renders.size()
        renders = renders.view(shape[0]*shape[1], shape[2], shape[3], shape[4])

        # pass the encoder and reshape render features from dimension (N*K)*D1 to N*(K*D1)
        render_feature = self.render_encoder(renders)
        render_feature = render_feature.view(render_feature.size(0), -1)
        render_feature = self.fc1(render_feature)
        if self.view_pooling:
            render_feature = render_feature.view(-1, self.render_number, self.render_feature_dim)
            render_feature = self.pooling(render_feature.permute(0, 2, 1)).squeeze()
        else:
            render_feature = render_feature.view(-1, self.render_number * self.render_feature_dim)

        real_feature = self.real_encoder(im)
        real_feature = real_feature.view(real_feature.size(0), -1)
        real_feature = self.fc2(real_feature)

        # concatenate the features obtained from two encoders into one feature of dimension N*(K*D1+D2)
        global_feature = torch.cat((render_feature, real_feature), 1)
        x = self.compress(global_feature)
        #out_azi = self.fc3(x)
        #out_ele = self.fc4(x)
        out_azi = self.fc4(x)
        out_ele = self.fc3(x)

        if self.all_angles:
            out_inp = self.fc5(x)
            return [out_azi, out_ele, out_inp]
        else:
            return [out_azi, out_ele]


class MultiViewNetMix(nn.Module):
    def __init__(self, render_number=12, render_feature_dim=256, real_feature_dim=1024, share_branch=False,
                 azi_classes=24, ele_classes=12, inp_classes=24, all_angles=True, channels=3,
                 features=64, input_dim=224, pretrained_resnet=False, res_pooling=True, res=18):
        super(MultiViewNetMix, self).__init__()
        self.render_feature_dim = render_feature_dim
        self.render_number = render_number
        self.all_angles = all_angles
        self.share_branch = share_branch

        ResNet = resnet.resnet18 if res == 18 else resnet.resnet34
        render_encoder = ResNet(input_channel=channels, num_classes=render_feature_dim, features=features)

        # if res_pooling is True, use the AvgPool layer in ResNet network
        last_layer = -1 if res_pooling else -2
        last_layer_dim = 1 if res_pooling else int(input_dim/32)**2

        self.render_encoder = nn.Sequential(*list(render_encoder.children())[:last_layer])
        self.fc1 = nn.Sequential(nn.Linear(last_layer_dim*features*8, render_feature_dim),
                                 nn.BatchNorm1d(render_feature_dim), nn.ReLU(inplace=True))

        real_encoder = ResNet(pretrained=pretrained_resnet, input_channel=3, num_classes=real_feature_dim, features=features)
        self.real_encoder = nn.Sequential(*list(real_encoder.children())[:last_layer])
        self.fc2 = nn.Sequential(nn.Linear(last_layer_dim*features*8, real_feature_dim),
                                 nn.BatchNorm1d(real_feature_dim), nn.ReLU(inplace=True))

        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.compress_cls = nn.Sequential(nn.Linear(render_number * render_feature_dim + real_feature_dim, 800),
                                      nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
        self.compress_reg = nn.Sequential(nn.Linear(render_number * render_feature_dim + real_feature_dim, 800),
                                      nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(200, azi_classes)
        self.fc4 = nn.Linear(200, ele_classes)
        self.fc5 = nn.Linear(200, inp_classes)
        self.fc6 = nn.Linear(200, 3)

    def forward(self, im, renders):
        # reshape render images from dimension N*K*C*H*W to (N*K)*C*H*W
        shape = renders.size()
        renders = renders.view(shape[0]*shape[1], shape[2], shape[3], shape[4])

        # pass the encoder and reshape render features from dimension (N*K)*D1 to N*(K*D1)
        render_feature = self.render_encoder(renders)
        render_feature = render_feature.view(render_feature.size(0), -1)
        render_feature = self.fc1(render_feature)
        render_feature = render_feature.view(-1, self.render_number * self.render_feature_dim)

        real_feature = self.real_encoder(im)
        real_feature = real_feature.view(real_feature.size(0), -1)
        real_feature = self.fc2(real_feature)

        # concatenate the features obtained from two encoders into one feature of dimension N*(K*D1+D2)
        global_feature = torch.cat((render_feature, real_feature), 1)
        x_cls = self.compress_cls(global_feature)
        x_reg = self.compress_cls(global_feature) if self.share_branch else self.compress_reg(global_feature)
        out_azi = self.fc3(x_cls)
        out_ele = self.fc4(x_cls)
        out_inp = self.fc5(x_cls)
        out_reg = self.fc6(x_reg)
        return [out_azi, out_ele, out_inp, out_reg]


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
