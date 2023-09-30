from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel

class SaliencyGradient:
    """
    SM
    """
    def __init__(self, model):
        self.model = model
        self.saliency = Saliency(model)

    def __call__(self, data, target):
        attribution_map = self.saliency.attribute(data, target=target, abs=False)
        return attribution_map.detach().cpu().numpy()