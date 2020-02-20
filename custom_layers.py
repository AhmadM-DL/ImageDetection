import torch.nn as nn

class DummyLayer(nn.Module):
    def __init__(self):
        super(DummyLayer, self).__init__()


class Detector(nn.Module):
    def __init__(self, anchors):
        super(Detector, self).__init__()
        self.anchors = anchors
