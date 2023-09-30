from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel

class DL:
    """
    DeepLift
    """
    def __init__(self, model):
        self.model = model
        self.saliency = DeepLift(model)

    def __call__(self, data, target):
        attribution_map = self.saliency.attribute(data, target=target, baselines=None)
        return attribution_map.detach().cpu().numpy()