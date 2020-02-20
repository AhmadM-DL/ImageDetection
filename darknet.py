import torch
import torch.nn as nn
import numpy as np
import custom_layers
import darknet_utils


def construct_cfg(config_file):
    """
    Build the network blocks using the configuration file.
    Pre-process it to form easy to manipulate using pytorch.
    """

    # Read and pre-process the configuration file
    config = open(config_file, 'r')
    file = config.read().split('\n')

    file = [line for line in file if len(line) > 0 and line[0] != '#']
    file = [line.lstrip().rstrip() for line in file]

    # Separate network blocks in a list
    network_blocks = []
    network_block = {}

    for x in file:
        if x[0] == '[':
            if len(network_block) != 0:
                network_blocks.append(network_block)
                network_block = {}
            network_block["type"] = x[1:-1].rstrip()
        else:
            entity, value = x.split('=')
            network_block[entity.rstrip()] = value.lstrip()
    network_blocks.append(network_block)

    return network_blocks


def construct_network_from_cfg(network_blocks):
    network_hyperparameters = network_blocks[0]
    modules = nn.ModuleList([])
    channels = 3
    filter_tracker = []

    for i, x in enumerate(network_blocks[1:]):
        seq_module = nn.Sequential()

        if x["type"] == "convolutional":

            filters = int(x["filters"])
            pad = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            padding = (kernel_size - 1) // 2 if pad else 0

            activation = x["activation"]

            try:
                bn = int(x["batch_normalize"])
                bias = False
            except:
                bn = 0
                bias = True

            conv = nn.Conv2d(channels, filters, kernel_size, stride, padding, bias=bias)
            seq_module.add_module("conv_{0}".format(i), conv)

            if bn:
                bn = nn.BatchNorm2d(filters)
                seq_module.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                seq_module.add_module("leaky_{0}".format(i), activn)

        elif x["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            seq_module.add_module("upsample_{}".format(i), upsample)

        elif x["type"] == "route":
            x['layers'] = x["layers"].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - i
            if end > 0:
                end = end - i

            route = custom_layers.DummyLayer()
            seq_module.add_module("route_{0}".format(i), route)
            if end < 0:
                filters = filter_tracker[i + start] + filter_tracker[i + end]
            else:
                filters = filter_tracker[i + start]

        elif x["type"] == "shortcut":
            shortcut = custom_layers.DummyLayer()
            seq_module.add_module("shortcut_{0}".format(i), shortcut)

        elif x["type"] == "yolo":
            anchors = x["anchors"].split(',')
            anchors = [int(a) for a in anchors]
            masks = x["mask"].split(',')
            masks = [int(a) for a in masks]
            anchors = [(anchors[j], anchors[j + 1]) for j in range(0, len(anchors), 2)]
            anchors = [anchors[j] for j in masks]
            detector_layer = custom_layers.Detector(anchors)

            seq_module.add_module("Detection_{0}".format(i), detector_layer)

        modules.append(seq_module)
        channels = filters
        filter_tracker.append(filters)
    return network_hyperparameters, modules


class DarkNet(nn.Module):

    def __init__(self, cfg_file):
        super(DarkNet, self).__init__()
        self.net_blocks = construct_cfg(cfg_file)
        self.network_hyperparameters, self.module_list = construct_network_from_cfg(self.net_blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x, CUDA):
        detections = []
        modules = self.net_blocks[1:]
        layer_outputs = {}

        written_output = 0

        # Iterate through each module
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            # Upsampling is basically a form of convolution
            if module_type == "convolutional" or module_type == "upsample":

                x = self.moduleList[i](x)
                layer_outputs[i] = x

            # Add outouts from previous layers to this layer
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                # If layer number is mentioned instead of its position relative to the the current layer
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = layer_outputs[i + (layers[0])]

                else:
                    # If layer number is mentioned instead of its position relative to the the current layer
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = layer_outputs[i + layers[0]]
                    map2 = layer_outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
                layer_outputs[i] = x

            # ShortCut is essentially residue from resnets
            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = layer_outputs[i - 1] + layer_outputs[i + from_]
                layer_outputs[i] = x

            elif module_type == 'yolo':

                anchors = self.moduleList[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.DNInfo["height"])

                # Get the number of classes
                num_classes = int(modules[i]["classes"])

                # Output the result
                x = x.data
                print("Size before transform => ", x.size())

                # Convert the output to 2D (batch x grids x bounding box attributes)
                x = darknet_utils.transformOutput(x, inp_dim, anchors, num_classes, CUDA)
                print("Size after transform => ", x.size())

                # If no detections were made
                if type(x) == int:
                    continue

                if not written_output:
                    detections = x
                    written_output = 1

                else:
                    detections = torch.cat((detections, x), 1)

                layer_outputs[i] = layer_outputs[i - 1]

        try:
            return detections
        except:
            return 0

    def load_weights(self, weight_file):

        fp = open(weight_file, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. Images seen

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        tracker = 0
        for i in range(len(self.moduleList)):
            module_type = self.netBlocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.moduleList[i]
                try:
                    batch_normalize = int(self.netBlocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv_part = model[0]

                if batch_normalize:
                    # Weights file Configuration=> bn bais->bn weights-> running mean-> running var
                    # The weights are arranged in the above mentioned order
                    bn_part = model[1]

                    bias_count = bn_part.bias.numel()

                    bn_bias = torch.from_numpy(weights[tracker:tracker + bias_count])
                    tracker += bias_count

                    bn_part_weights = torch.from_numpy(weights[tracker: tracker + bias_count])
                    tracker += bias_count

                    bn_part_running_mean = torch.from_numpy(weights[tracker: tracker + bias_count])
                    tracker += bias_count

                    bn_part_running_var = torch.from_numpy(weights[tracker: tracker + bias_count])
                    tracker += bias_count

                    bn_bias = bn_bias.view_as(bn_part.bias.data)
                    bn_part_weights = bn_part_weights.view_as(bn_part.weight.data)
                    bn_part_running_mean = bn_part_running_mean.view_as(bn_part.running_mean)
                    bn_part_running_var = bn_part_running_var.view_as(bn_part.running_var)

                    bn_part.bias.data.copy_(bn_bias)
                    bn_part.weight.data.copy_(bn_part_weights)
                    bn_part.running_mean.copy_(bn_part_running_mean)
                    bn_part.running_var.copy_(bn_part_running_var)

                else:
                    bias_count = conv_part.bias.numel()

                    conv_bias = torch.from_numpy(weights[tracker: tracker + bias_count])
                    tracker = tracker + bias_count

                    conv_bias = conv_bias.view_as(conv_part.bias.data)

                    conv_part.bias.data.copy_(conv_bias)

                weight_file = conv_part.weight.numel()

                conv_weight = torch.from_numpy(weights[tracker:tracker + weight_file])
                tracker = tracker + weight_file

                conv_weight = conv_weight.view_as(conv_part.weight.data)
                conv_part.weight.data.copy_(conv_weight)


# def compute_loss(output, target):
#
#     if x.is_cuda:
#         self.mse_loss = self.mse_loss.cuda()
#         self.bce_loss = self.bce_loss.cuda()
#         self.ce_loss = self.ce_loss.cuda()
#
#     nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
#         pred_boxes=pred_boxes.cpu().data,
#         pred_conf=pred_conf.cpu().data,
#         pred_cls=pred_cls.cpu().data,
#         target=targets.cpu().data,
#         anchors=scaled_anchors.cpu().data,
#         num_anchors=nA,
#         num_classes=self.num_classes,
#         grid_size=nG,
#         ignore_thres=self.ignore_thres,
#         img_dim=self.image_dim,
#     )
#
#     nProposals = int((pred_conf > 0.5).sum().item())
#     recall = float(nCorrect / nGT) if nGT else 1
#     precision = 0
#     if nProposals > 0:
#         precision = float(nCorrect / nProposals)
#
#     # Handle masks
#     mask = Variable(mask.type(ByteTensor))
#     conf_mask = Variable(conf_mask.type(ByteTensor))
#
#     # Handle target variables
#     tx = Variable(tx.type(FloatTensor), requires_grad=False)
#     ty = Variable(ty.type(FloatTensor), requires_grad=False)
#     tw = Variable(tw.type(FloatTensor), requires_grad=False)
#     th = Variable(th.type(FloatTensor), requires_grad=False)
#     tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
#     tcls = Variable(tcls.type(LongTensor), requires_grad=False)
#
#     # Get conf mask where gt and where there is no gt
#     conf_mask_true = mask
#     conf_mask_false = conf_mask - mask
#
#     # Mask outputs to ignore non-existing objects
#     loss_x = self.mse_loss(x[mask], tx[mask])
#     loss_y = self.mse_loss(y[mask], ty[mask])
#     loss_w = self.mse_loss(w[mask], tw[mask])
#     loss_h = self.mse_loss(h[mask], th[mask])
#     loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
#         pred_conf[conf_mask_true], tconf[conf_mask_true]
#     )
#     loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
#     loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
#
#     return (
#         loss,
#         loss_x.item(),
#         loss_y.item(),
#         loss_w.item(),
#         loss_h.item(),
#         loss_conf.item(),
#         loss_cls.item(),
#         recall,
#         precision,
#     )

# def train(image_folder, model_config_path, data_config_path, weights_path, classes_path, checkpoint_dir, device,
#           batch_size=16, n_epochs=20, conf_thres=0.8, nms_thres=0.4, n_cpu=0, img_size=416, checkpoint_interval=1):
#     os.makedirs(checkpoint_dir, exist_ok=True)
#
#     classes = utils.load_classes(classes_path)
#
#     # Get data configuration
#     data_config = utils.parse_data_config(data_config_path)
#     train_path = data_config["train"]
#
#     # Get hyper parameters
#     hyperparams = utils.parse_model_config(model_config_path)[0]
#
#     learning_rate = float(hyperparams["learning_rate"])
#     momentum = float(hyperparams["momentum"])
#     decay = float(hyperparams["decay"])
#     burn_in = int(hyperparams["burn_in"])
#
#     # Initiate model
#     model = Darknet(model_config_path)
#     model.load_weights(weights_path)
#
#     model = model.to(device)
#
#     model.train()
#
#     # Get dataloader
#     dataloader = torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size,
#                                              shuffle=False, num_workers=n_cpu)
#
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
#
#     for epoch in range(n_epochs):
#         for batch_i, (_, imgs, targets) in enumerate(dataloader):
#             imgs = imgs.to(device)
#             targets = targets.to(device)
#
#             optimizer.zero_grad()
#
#             loss = model(imgs, targets)
#
#             loss.backward()
#             optimizer.step()
#
#             print(
#                 "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
#                 % (
#                     epoch,
#                     n_epochs,
#                     batch_i,
#                     len(dataloader),
#                     model.losses["x"],
#                     model.losses["y"],
#                     model.losses["w"],
#                     model.losses["h"],
#                     model.losses["conf"],
#                     model.losses["cls"],
#                     loss.item(),
#                     model.losses["recall"],
#                     model.losses["precision"],
#                 )
#             )
#
#             model.seen += imgs.size(0)
#
#         if epoch % opt.checkpoint_interval == 0:
#             model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))

