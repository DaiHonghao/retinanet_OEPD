import torch.nn as nn
import torch.nn.functional
import torch
import math
import torch.utils.model_zoo as model_zoo
import time
from torchvision.ops import nms
# from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
# from retinanet.anchors import Anchors
# from retinanet import losses
from .utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from .anchors import Anchors
from . import losses
from torch.utils.tensorboard import SummaryWriter

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_1_hm = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2_hm = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1_hm = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2_hm = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1_hm = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2_hm = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P2_1_hm = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2_hm = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        channel = 64
        self.query_enlarge = nn.Upsample(size=None, scale_factor=2, mode="nearest")

        self.heat_map_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, 1,
                      kernel_size=1, stride=1, padding=0))

        self.key_point = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel, 1,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, inputs):
        heatmap = {}
        C2, C3, C4, C5 = inputs

        sw = SummaryWriter('./work_dir')

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        # layer5
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P5_x_hm = self.P5_1_hm(C5.clone().detach())
        P5_upsampled_x_hm = self.P5_upsampled(P5_x_hm)
        P5_x_hm = self.P5_2_hm(P5_x_hm)

        hm = self.heat_map_head(P5_x_hm).sigmoid()
        a = torch.nn.functional.interpolate(hm, size=[200, 200])
        sw.add_images('5', a, 1)
        kp = self.key_point(P5_x).sigmoid()
        heatmap['5'] = hm
        next_query = kp + torch.tensor(1.)
        query_map = self.query_enlarge(next_query)

        # layer4
        P4_x = self.P4_1(C4)
        P5_upsampled_x = torch.nn.functional.interpolate(P5_upsampled_x, size=[int(P4_x.shape[2]), int(P4_x.shape[3])])
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P4_x_hm = self.P4_1_hm(C4.clone().detach())
        P5_upsampled_x_hm = torch.nn.functional.interpolate(P5_upsampled_x_hm, size=[int(P4_x.shape[2]), int(P4_x.shape[3])])
        P4_x_hm = P5_upsampled_x_hm + P4_x_hm
        P4_upsampled_x_hm = self.P4_upsampled(P4_x_hm)
        P4_x_hm = self.P4_2_hm(P4_x_hm)

        query_map = self.fix_shape(P4_x, query_map)
        P4_x = torch.mul(P4_x, query_map).to(torch.float32)
        P4_x = P4_x + torch.nn.functional.interpolate(P5_x, size=[int(P4_x.shape[2]), int(P4_x.shape[3])])

        hm = self.heat_map_head(P4_x_hm).sigmoid()
        kp = self.key_point(P4_x).sigmoid()
        a = a + torch.nn.functional.interpolate(hm, size=[200, 200])
        sw.add_images('4', torch.nn.functional.interpolate(hm, scale_factor=4), 1)
        heatmap['4'] = hm
        next_query = kp
        next_query = next_query + query_map + torch.tensor(1.)
        query_map = self.query_enlarge(next_query)

        # layer3
        P3_x = self.P3_1(C3)
        P4_upsampled_x = torch.nn.functional.interpolate(P4_upsampled_x, size=[int(P3_x.shape[2]), int(P3_x.shape[3])])
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P3_x_hm = self.P3_1_hm(C3.clone().detach())
        P4_upsampled_x_hm = torch.nn.functional.interpolate(P4_upsampled_x_hm, size=[int(P3_x_hm.shape[2]), int(P3_x_hm.shape[3])])
        P3_x_hm = P4_upsampled_x_hm + P3_x_hm
        P3_upsampled_x_hm = self.P3_upsampled(P3_x_hm)
        P3_x_hm = self.P3_2_hm(P3_x_hm)

        query_map = self.fix_shape(P3_x, query_map)
        P3_x = torch.mul(P3_x, query_map).to(torch.float32)
        P3_x = P3_x + torch.nn.functional.interpolate(P4_x, size=[int(P3_x.shape[2]), int(P3_x.shape[3])])

        hm = self.heat_map_head(P3_x_hm).sigmoid()
        kp = self.key_point(P3_x).sigmoid()
        a = a + torch.nn.functional.interpolate(hm, size=[200, 200])
        sw.add_images('3', torch.nn.functional.interpolate(hm, scale_factor=2), 1)

        heatmap['3'] = hm
        next_query = kp
        next_query = next_query + query_map + torch.tensor(1.)
        query_map = self.query_enlarge(next_query)

        P2_x = self.P2_1(C2)
        P3_upsampled_x = torch.nn.functional.interpolate(P3_upsampled_x, size=[int(P2_x.shape[2]), int(P2_x.shape[3])])
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P2_x_hm = self.P2_1_hm(C2.clone().detach())
        P3_upsampled_x_hm = torch.nn.functional.interpolate(P3_upsampled_x_hm, size=[int(P2_x_hm.shape[2]), int(P2_x_hm.shape[3])])
        P2_x_hm = P3_upsampled_x_hm + P2_x_hm
        P2_x_hm = self.P2_2_hm(P2_x_hm)

        query_map = self.fix_shape(P2_x, query_map)
        P2_x = torch.mul(P2_x, query_map).to(torch.float32)
        P2_x = P2_x + torch.nn.functional.interpolate(P3_x, size=[int(P2_x.shape[2]), int(P2_x.shape[3])])

        heatmap['2'] = hm

        return [P2_x, P3_x, P4_x, P5_x], heatmap, P7_x

    def fix_shape(self, input, query_map):
        if input.shape == query_map.shape:
            return query_map
        else:
            src_bs, src_chan, src_h, src_w = input.shape
            quer_bs, quer_chan, quer_h, quer_w = query_map.shape
            if src_h != quer_h:
                if src_h < quer_h:
                    query_map = query_map[:, :, 0:src_h, :]
                elif src_h > quer_h:
                    temp = torch.ones(quer_bs, quer_chan, src_h - quer_h, quer_w)
                    temp = temp * 0.4
                    query_map = torch.cat((query_map, temp), dim=2)
            if src_w != quer_w:
                if src_w < quer_w:
                    query_map = query_map[:, :, :, 0:src_w]
                elif src_w > quer_w:
                    temp = torch.ones(quer_bs, quer_chan, src_h, src_w - quer_w)
                    temp = temp * 0.4
                    query_map = torch.cat((query_map, temp), dim=3)
            return query_map


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=21, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=21, num_classes=8, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.combine = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels,
                         self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels,
                         self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        self.heatmapLoss = losses.loss_heatmap()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs, if_trans):
        device = torch.device('cuda')
        if self.training:
            img_batch, annotations, batch_hm = inputs
        else:
            img_batch, batch_hm = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features, heatmap, bg = self.fpn([x1, x2, x3, x4])
        # layer = ['2', '3', '4', '5']
        # heatmap2 = {}
        # if self.training:
        #     for i in layer:
        #         heatmap2[i] = torch.cat([v[i].permute(2, 0, 1).unsqueeze(0) for v in batch_hm], dim=0)
        # else:
        #     for i in layer:
        #         heatmap2[i] = batch_hm[0][i].permute(2, 0, 1).unsqueeze(0)
        #         # heatmap[i] = torch.cat([v[i].permute(2, 0, 1).unsqueeze(0) for v in batch_hm], dim=0)

        # if if_trans:
        #     if self.training:
        #         for i in layer:
        #             heatmap[i] = torch.cat([v[i].permute(2, 0, 1).unsqueeze(0) for v in batch_hm], dim=0)
        #     else:
        #         for i in layer:
        #             # heatmap[i] = batch_hm[i].permute(2, 0, 1).unsqueeze(0)
        #             heatmap[i] = torch.cat([v[i].permute(2, 0, 1).unsqueeze(0) for v in batch_hm], dim=0)
        #

        # if self.training:
        #     if not if_trans:
        #         return self.heatmapLoss(heatmap, batch_hm)
        #
        # for feature_num in range(len(features)):
        #     features[feature_num] = torch.mul(features[feature_num], torch.ceil(heatmap[str(feature_num+2)]-0.5))
        # feature [bs, 256, h, w]
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        # need_to_detect_image = []
        need_to_detect_box = []
        per_layer_num = []

        get_patch_nums = [220, 10, 1, 1]

        init_layer = 0
        focalmap = heatmap['2'] + torch.nn.functional.interpolate(heatmap['3'], scale_factor=2) + torch.nn.functional.interpolate(heatmap['4'], scale_factor=4)
        focalmap = torch.clamp(focalmap, 2, 3) - 2
        # [bbs, c, h, w]
        patch_image, batch_weight, per_batch_num, akaaa = self.get_patch(focalmap,
                                                                         features[0], init_layer,
                                                                         nums=get_patch_nums[0],
                                                                         patch_size=[4, 4], bg=bg)

        # need_to_detect_image.append(patch_image)
        need_to_detect_box.append(batch_weight)
        per_layer_num.append(per_batch_num)
        init_layer = per_batch_num[-1]

        # for input_layer in range(len(features)-2):
        #     # [bbs, c, h, w]
        #     patch_image, batch_weight, per_batch_num, akaaa = self.get_patch(heatmap[str(input_layer + 2)],
        #                                                               features[input_layer], init_layer,
        #                                                               nums=get_patch_nums[input_layer],
        #                                                               patch_size=[8, 8], bg=bg)
        #
        #     # need_to_detect_image.append(patch_image)
        #     need_to_detect_box.append(batch_weight)
        #     per_layer_num.append(per_batch_num)
        #     init_layer = per_batch_num[-1]

        # feature [bs, 256, h, w]
        # print('debug')
        # feature = torch.cat([v for v in need_to_detect_image], dim=0)
        box = torch.cat([v for v in need_to_detect_box], dim=0)
        # # a = time.time()
        # regression = self.regressionModel(feature)
        #
        # classification = self.classificationModel(feature)
        #
        # anchors = self.anchors(feature)
        # # b = time.time()
        # # print('time', b-a)
        # all_batch_regression = []
        # all_batch_classification = []
        all_box = []
        for batch_num in range(len(per_layer_num[0])-1):
            # one_batch_regression = []
            # one_batch_classification = []
            one_box = []
            for layer in per_layer_num:
                # one_batch_regression.append(regression[layer[batch_num]:layer[batch_num+1], :, :].unsqueeze(0).flatten(1, 2))
                # one_batch_classification.append(classification[layer[batch_num]:layer[batch_num + 1], :, :].unsqueeze(0).flatten(1, 2))
                one_box.append(box[layer[batch_num]:layer[batch_num + 1], :].unsqueeze(0))
            # all_batch_regression.append(torch.cat([v for v in one_batch_regression], dim=1))
            # all_batch_classification.append(torch.cat([v for v in one_batch_classification], dim=1))
            all_box.append(torch.cat([v for v in one_box], dim=1))
        #
        # regression = torch.cat([v for v in all_batch_regression], dim=0)
        # classification = torch.cat([v for v in all_batch_classification], dim=0)
        box = torch.cat([v for v in all_box], dim=0)
        # scale[4, 62]
        scale = 800 / box[:, :, 4]
        bf_box = box[:, :, 0:4] * scale.unsqueeze(2)

        # a = 1
        # box = box[:, :, 0:4]
        #
        # # anchors[1, 1296, 4], box[4, 62, 4]
        # bs, box_num, _ = box.shape
        # an_num = int(anchors.shape[1])
        # anchors = anchors.repeat(int(bs), int(box_num), 1)
        # box = box.repeat(1, 1, an_num)
        # box = box.reshape(int(bs), (int(box_num)*an_num), 4).to(device)
        # scale = scale.unsqueeze(dim=2)
        # scale = scale.repeat(1, 1, an_num).reshape(int(bs), (int(box_num)*an_num), -1).to(device)
        # box = box * scale
        # anchors = anchors * scale
        # anchors[:, :, 0] = anchors[:, :, 0] + box[:, :, 0]
        # anchors[:, :, 2] = anchors[:, :, 2] + box[:, :, 0]
        # anchors[:, :, 1] = anchors[:, :, 1] + box[:, :, 1]
        # anchors[:, :, 3] = anchors[:, :, 3] + box[:, :, 1]
        # # # debug
        # # /media/azach-3204/6C367DB2367D7DC0/dhh/duibi/wh/pytorch-retinanet-master/pytorch-retinanet-master/work_dir
        # # # 'E:/my_code/pytorch-retinanet-master/pytorch-retinanet-master/work_dir'
        # print('debug')
        sw = SummaryWriter('./work_dir')
        or_image = img_batch
        # or_image = or_image.repeat(3, 1, 1)
        # or_image = batch_hm[0]['2'].permute(2, 0, 1)
        pred_image = heatmap['2']
        #
        # # cnn = anchors.shape[1]
        # # for i in range(int(cnn)):
        # #     xmin = int(anchors[0, i, 0])
        # #     ymin = int(anchors[0, i, 1])
        # #     xmax = int(anchors[0, i, 2])
        # #     ymax = int(anchors[0, i, 3])
        # #     h = ymax - ymin
        # #     w = xmax - xmin
        # #     color = [0, 0, 1]
        # #     dep = 1
        # #     or_image = self.draw_reg(or_image, ymin, xmin, h, w, dep, color)
        # #
        bnn = bf_box.shape[1]
        # or_image = or_image.repeat(3, 1, 1)
        # or_image = torch.nn.functional.interpolate(or_image.unsqueeze(0), scale_factor=4)
        pred_image = pred_image[0, :, :, :]
        pred_image = pred_image.unsqueeze(0)
        pred_image = torch.nn.functional.interpolate(pred_image, size=[800, 800])
        pred_image = pred_image.repeat(1, 3, 1, 1)
        pred_image = pred_image[0, :, :, :]
        for i in range(int(bnn)):
            xmin = int(bf_box[0, i, 0]) if int(bf_box[0, i, 0]) >=0 else 0
            ymin = int(bf_box[0, i, 1]) if int(bf_box[0, i, 1]) >=0 else 0
            xmax = int(bf_box[0, i, 2])
            ymax = int(bf_box[0, i, 3])
            h = ymax - ymin
            w = xmax - xmin
            color = [0, 1, 0]
            dep = 2
            pred_image = self.draw_reg(pred_image, ymin, xmin, h, w, dep, color)
        pred_image = pred_image.unsqueeze(0)
        sw.add_images('orimage', or_image, 1)
        sw.add_images('patch8', pred_image, 1)

        # # ann = annotations.shape[1]
        # # for i in range(int(ann)):
        # #     xmin = int(annotations[0, i, 0])
        # #     ymin = int(annotations[0, i, 1])
        # #     xmax = int(annotations[0, i, 2])
        # #     ymax = int(annotations[0, i, 3])
        # #     h = ymax - ymin
        # #     w = xmax - xmin
        # #     color = [0, 1, 1]
        # #     dep = 2
        # #     or_image = self.draw_reg(or_image, ymin, xmin, h, w, dep, color)
        # #
        # #
        # # classification_loss, regression_loss, hm_loss, useful_anchors = self.focalLoss(classification, regression, anchors, annotations, bf_box, k=(self.heatmapLoss(heatmap, batch_hm)))
        # #
        # # for an in useful_anchors:
        # #     xmin = int(an[0])
        # #     ymin = int(an[1])
        # #     xmax = int(an[2])
        # #     ymax = int(an[3])
        # #     yc = int((ymax + ymin)/2)
        # #     xc = int((xmax + xmin)/2)
        # #     h = ymax - ymin
        # #     w = xmax - xmin
        # #     color = [0, 1, 1]
        # #     dep = 2
        # #     pred_image = self.draw_reg(pred_image, yc, xc, 2, 2, dep, color)
        # #
        # # ann = annotations.shape[1]
        # # for i in range(int(ann)):
        # #     xmin = int(annotations[0, i, 0])
        # #     ymin = int(annotations[0, i, 1])
        # #     xmax = int(annotations[0, i, 2])
        # #     ymax = int(annotations[0, i, 3])
        # #     h = ymax - ymin
        # #     w = xmax - xmin
        # #     color = [1, 0, 0]
        # #     dep = 2
        # #     pred_image = self.draw_reg(pred_image, ymin, xmin, h, w, dep, color)
        # #
        # # pred_image = pred_image.unsqueeze(0)
        # # or_image = or_image.unsqueeze(0)
        # # if if_trans:
        # #     ak = 1
        # # else:
        # #     ak = 2
        # #
        # # or_image = torch.cat((or_image, pred_image), dim=0)
        # # sw.add_images('patch8', or_image, ak)
        # sw = SummaryWriter('./work_dir')

        # pred_image1 = heatmap['2']
        # pred_image2 = heatmap['3']
        # pred_image3 = heatmap['4']
        # pred_image2 = torch.nn.functional.interpolate(pred_image2,scale_factor=2)
        # pred_image3 = torch.nn.functional.interpolate(pred_image3, scale_factor=4)
        # pred_image = pred_image1 + pred_image2 + pred_image3
        # pred_image3 = torch.clamp(pred_image, 0, 1)
        # pred_image2 = torch.clamp(pred_image, 1, 2) - 1
        # pred_image1 = torch.clamp(pred_image, 2, 3) - 2
        # pred_image3 = pred_image3 - pred_image2 - pred_image1
        # pred_image3 = torch.clamp(pred_image3, 0, 1)
        # pred_image2 = pred_image2 - pred_image1
        # pred_image2 = torch.clamp(pred_image2, 0, 1)
        # sw.add_images('2t', pred_image1, 1)
        # pred_image1 = torch.cat((pred_image1, pred_image2, pred_image3), dim=1)
        # sw.add_images("g2", batch_hm[0]["2"].permute(2, 0, 1).unsqueeze(0), 1)
        # sw.add_images('1', pred_image1, 1)
        #
        # import pdb
        # pdb.set_trace()
        # pred_image2 = heatmap['3']
        # pred_image3 = heatmap['4']
        # pred_image2 = torch.nn.functional.interpolate(pred_image2,scale_factor=2)
        # pred_image3 = torch.nn.functional.interpolate(pred_image3, scale_factor=4)
        # pred_image5 = heatmap['5']
        # pred_image5 = torch.nn.functional.interpolate(pred_image5, scale_factor=8)
        # pred_image1 = heatmap['2']
        # pred_image1 = torch.nn.functional.interpolate(pred_image1, scale_factor=8)
        # sw.add_images('2', pred_image1, 1)
        # sw.add_images('5', pred_image5, 1)
        # sw.add_images('3', pred_image2, 1)
        # sw.add_images('4', pred_image3, 1)

        # # aaaaaaaaaaaaaaaaaaaaaaa
        # import pdb
        # pdb.set_trace()
        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations, bf_box, k=(self.heatmapLoss(heatmap, batch_hm)))
            # if if_trans:
            #     return self.focalLoss(classification, regression, anchors, annotations)
            # else:
            #     return self.heatmapLoss(heatmap, batch_hm)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, bf_box, akaaa]

    def draw_reg(self, or_image, ymin, xmin, h, w, dep, color):
        c, yax, xax = or_image.shape
        yax = int(yax)
        xax = int(xax)
        ymin_h = ymin + h if ymin + h < yax else yax
        xmin_w = xmin + w if xmin + w < xax else xax
        or_image[0, ymin:ymin_h, xmin:xmin + dep] = color[0]
        or_image[1, ymin:ymin_h, xmin:xmin + dep] = color[1]
        or_image[2, ymin:ymin_h, xmin:xmin + dep] = color[2]
        or_image[0, ymin:ymin_h, xmin_w:xmin + dep + w] = color[0]
        or_image[1, ymin:ymin_h, xmin_w:xmin + dep + w] = color[1]
        or_image[2, ymin:ymin_h, xmin_w:xmin + dep + w] = color[2]
        or_image[0, ymin:ymin + dep, xmin:xmin_w] = color[0]
        or_image[1, ymin:ymin + dep, xmin:xmin_w] = color[1]
        or_image[2, ymin:ymin + dep, xmin:xmin_w] = color[2]
        or_image[0, ymin_h:ymin + dep + h, xmin:xmin_w] = color[0]
        or_image[1, ymin_h:ymin + dep + h, xmin:xmin_w] = color[1]
        or_image[2, ymin_h:ymin + dep + h, xmin:xmin_w] = color[2]
        return or_image

    def get_patch(self, hm, src, init_layer, nums, patch_size, bg):
        sw = SummaryWriter('./work_dir')
        device = torch.device('cuda')
        bf_hm = hm.clone().detach()
        visu_hm = hm.clone().detach()
        visu_hm2 = hm.clone().detach()
        hb, hc, hh, hw = bf_hm.shape
        bf_hm = bf_hm.flatten(2, 3)
        return_weight = []
        return_image = []
        batch_sum = init_layer
        per_batch_num = [batch_sum]
        for i in range(hb):
            flag = True
            num = 0
            per_batch_image = []
            per_batch_patch = []
            # find all patch in one img
            num_for = 0
            visu_hm_out2 = visu_hm2[i, :, :, :]
            visu_hm_out2 = visu_hm_out2.repeat(3, 1, 1)
            while flag and num < nums:
                num_for = num_for + 1
                num = num + 1
                value, index = torch.max(bf_hm, dim=2)
                flag = False
                # 则在通道0上，最大值为index[0]
                patch_inform = []
                if value[i][0] > 0.5:
                    flag = True
                    x_int = index[i][0] % hw
                    y_int = int(index[i][0] / hh)
                    bf_hm = bf_hm.reshape(hb, hc, hh, hw)
                    nh = torch.tensor(patch_size[0])
                    nw = torch.tensor(patch_size[1])
                    nh = torch.clamp(nh, 0.001, hh)
                    nw = torch.clamp(nw, 0.001, hw)
                    xmin, xmax = self.c_to_min_max(x_int, nw, hw)
                    xmin, xmax = self.fix_min_max(xmin, xmax, hw)
                    ymin, ymax = self.c_to_min_max(y_int, nh, hh)
                    ymin, ymax = self.fix_min_max(ymin, ymax, hh)
                    if_new = True
                    old_inx = 0
                    # check and combine
                    visu_hm_out2 = self.draw_reg(visu_hm_out2, ymin, xmin, nh, nw, 2, [1, 0, 0])
                    visu_hm_out2 = visu_hm_out2.unsqueeze(0)
                    sw.add_images('before', visu_hm_out2, num_for)
                    visu_hm_out2 = visu_hm_out2.squeeze(0)
                    for patchlen in range(len(per_batch_patch)):
                        before_xmin = per_batch_patch[patchlen][0]
                        before_xmax = per_batch_patch[patchlen][1]
                        before_ymin = per_batch_patch[patchlen][2]
                        before_ymax = per_batch_patch[patchlen][3]
                        co_xmin = before_xmin if before_xmin > xmin else xmin
                        co_xmax = before_xmax if before_xmax < xmax else xmax
                        co_ymin = before_ymin if before_ymin > ymin else ymin
                        co_ymax = before_ymax if before_ymax < ymax else ymax
                        if (co_xmax - co_xmin) > 0 and (co_ymax - co_ymin > 0):
                            max_xmax = xmax if xmax > before_xmax else before_xmax
                            min_xmin = xmin if xmin < before_xmin else before_xmin
                            max_ymax = ymax if ymax > before_ymax else before_ymax
                            min_ymin = ymin if ymin < before_ymin else before_ymin
                            all_s = (max_xmax - min_xmin) * (max_ymax - min_ymin)
                            stand_s = patch_size[1] * patch_size[0]
                            co_h = co_ymax - co_ymin
                            co_w = co_xmax - co_xmin
                            co_s = co_h * co_w
                            if co_s / stand_s > 0:
                                if_new = False
                                xmin = min_xmin
                                xmax = max_xmax
                                ymin = min_ymin
                                ymax = max_ymax
                                old_inx = patchlen
                                break

                    patch_inform.append(xmin)
                    patch_inform.append(xmax)
                    patch_inform.append(ymin)
                    patch_inform.append(ymax)
                    patch_inform.append(hh)
                    # [1,:,h,w]
                    # [B, C, W, H]
                    patch_image = src[i, :, ymin:ymax, xmin:xmax].unsqueeze(0)
                    # 将patch缩放成相同大小
                    patch_image = torch.nn.functional.interpolate(patch_image, size=[patch_size[0], patch_size[1]])

                    inter_bg = torch.nn.functional.interpolate(bg[i, :, :, :].unsqueeze(0),
                                                                   size=[src[i].shape[1], src[i].shape[2]])
                    inter_bg = inter_bg[0, :, ymin:ymax, xmin:xmax].unsqueeze(0)
                    inter_bg = torch.nn.functional.interpolate(inter_bg, size=[patch_size[0], patch_size[1]])
                    patch_image = patch_image + inter_bg
                    # patch_image = torch.cat((patch_image, inter_bg), dim=1)
                    # patch_image = self.combine(patch_image)
                    if if_new:
                        per_batch_image.append(patch_image)
                        per_batch_patch.append(patch_inform)
                    else:
                        per_batch_image[old_inx] = patch_image
                        per_batch_patch[old_inx] = patch_inform
                        num = num - 1
                    bf_hm[i, 0, int(ymin):int(ymax), int(xmin):int(xmax)] = 0
                    bf_hm = bf_hm.flatten(2, 3)

            for ii in range(len(per_batch_patch)):
                xmin = per_batch_patch[ii][0]
                xmax = per_batch_patch[ii][1]
                ymin = per_batch_patch[ii][2]
                ymax = per_batch_patch[ii][3]
                x_c = int((xmax + xmin) / 2)
                y_c = int((ymax + ymin) / 2)
                h = int((ymax - ymin))
                w = int((xmax - xmin))
                nxmin = x_c - w
                nxmax = x_c + w
                nymin = y_c - h
                nymax = y_c + h
                per_batch_patch[ii][0] = nxmin
                per_batch_patch[ii][1] = nxmax
                per_batch_patch[ii][2] = nymin
                per_batch_patch[ii][3] = nymax

            refine = []
            for ii in range(len(per_batch_patch)):
                new_fine = []
                xmin = per_batch_patch[ii][0]
                xmax = per_batch_patch[ii][1]
                ymin = per_batch_patch[ii][2]
                ymax = per_batch_patch[ii][3]
                for patchlen in range(len(per_batch_patch)):
                    before_xmin = per_batch_patch[patchlen][0]
                    before_xmax = per_batch_patch[patchlen][1]
                    before_ymin = per_batch_patch[patchlen][2]
                    before_ymax = per_batch_patch[patchlen][3]
                    co_xmin = before_xmin if before_xmin > xmin else xmin
                    co_xmax = before_xmax if before_xmax < xmax else xmax
                    co_ymin = before_ymin if before_ymin > ymin else ymin
                    co_ymax = before_ymax if before_ymax < ymax else ymax
                    if (co_xmax - co_xmin) >= 0 and (co_ymax - co_ymin >= 0):
                        xmin = xmin if xmin < before_xmin else before_xmin
                        xmax = xmax if xmax > before_xmax else before_xmax
                        ymin = ymin if ymin < before_ymin else before_ymin
                        ymax = ymax if ymax > before_ymax else before_ymax
                new_fine.append(xmin)
                new_fine.append(xmax)
                new_fine.append(ymin)
                new_fine.append(ymax)
                new_fine.append(hh)
                refine.append(new_fine)

            per_batch_patch = refine
            visu_hm_out2 = visu_hm[i, :, :, :]
            visu_hm_out2 = visu_hm_out2.repeat(3, 1, 1)
            visu_hm_out2 = visu_hm_out2.unsqueeze(0)
            sw.add_images('ori', visu_hm_out2, num_for)
            visu_hm_out2 = visu_hm_out2.squeeze(0)
            for patchlen in range(len(per_batch_patch)):
                before_xmin = per_batch_patch[patchlen][0]
                before_xmax = per_batch_patch[patchlen][1]
                before_ymin = per_batch_patch[patchlen][2]
                before_ymax = per_batch_patch[patchlen][3]
                x_c = int((before_xmax + before_xmin)/2)
                y_c = int((before_ymax + before_ymin)/2)
                h = int((before_ymax - before_ymin))
                w = int((before_xmax - before_xmin))
                ymin_c = int(y_c - (h/2)) if int(y_c - (h/2)) > 0 else 0
                xmin_c = int(x_c - (w/2)) if int(x_c - (w/2)) > 0 else 0

                visu_hm_out2 = self.draw_reg(visu_hm_out2, ymin_c, xmin_c, h, w, 2, [1, 1, 0])
            visu_hm_out2 = visu_hm_out2.unsqueeze(0)
            sw.add_images('after', visu_hm_out2, num_for)
            visu_hm_out2 = visu_hm_out2.squeeze(0)
            # import pdb
            # pdb.set_trace()
            akaaa = per_batch_patch.__len__()
            if per_batch_image.__len__() < nums:
                fix = nums - per_batch_image.__len__()
                for fix_zero in range(fix):
                    patch_inform = []
                    _, c_im, _, _ = src.shape
                    zero_image = torch.zeros(1, c_im, patch_size[0], patch_size[1]).to(device)
                    per_batch_image.append(zero_image)
                    patch_inform.append(0)
                    patch_inform.append(1)
                    patch_inform.append(0)
                    patch_inform.append(1)
                    patch_inform.append(800)
                    per_batch_patch.append(patch_inform)

            per_batch_patch = torch.tensor(per_batch_patch)

            per_image_all = torch.cat([v for v in per_batch_image], dim=0)
            per_box_all = self.box_xxyy_to_xyxy(per_batch_patch)
            # print('per_image_all', per_image_all.shape)
            batch_sum = batch_sum + int(per_image_all.shape[0])
            per_batch_num.append(batch_sum)

            return_image.append(per_image_all)
            return_weight.append(per_box_all)
            value, iind = torch.max(bf_hm, dim=2)

        return_all_image = torch.cat([v for v in return_image], dim=0)
        # print('return_all_image', return_all_image.shape)
        # for i in range(len(per_batch_num)-1):
        #     print('aaaa', return_all_image[per_batch_num[i]:per_batch_num[i+1], :, :, :].shape)
        return_all_weight = torch.cat([v for v in return_weight], dim=0)
        # print('return_all_weight', return_all_weight.shape)
        # print('per_batch_num', per_batch_num)
        # IMG[HW, BS, C], POS[HW, BS, C], MASK[BS, HW], POINT[8, C], WEIGHT[8]

        return return_all_image, return_all_weight, per_batch_num, akaaa

    def c_to_min_max(self, c, nw, hw):
        min = int((c - nw / 2) if (c - nw / 2) - 1 > 0 else 0)
        max = int((c + nw / 2) if (c + nw / 2) + 1 < hw else hw)
        return min, max

    def fix_min_max(self, min, max, hh):
        if max == min:
            max = max + 1
            min = min - 1
            if max > hh - 1:
                max = max - 1
                min = min - 1
            if min < 0:
                min = min + 1
                max = max + 1
        if max < min:
            print('warning max < min')
            exit()
        return min, max

    def box_xyxy_to_cxcywh(self, x):
        x0, x1, y0, y1, size = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0), size]
        return torch.stack(b, dim=-1)

    def box_xxyy_to_xyxy(self, x):
        x0, x1, y0, y1, size = x.unbind(-1)
        b = [x0, y0, x1, y1, size]
        return torch.stack(b, dim=-1)


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
