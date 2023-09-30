from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel

class SmoothGradient:
    """
    SG
    """
    def __init__(self, model, stdevs=0.15):
        self.model = model
        self.saliency = NoiseTunnel(Saliency(model))
        self.stdevs = stdevs

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target,
                                                  nt_samples = gradient_steps,
                                                  stdevs=self.stdevs,
                                                  abs=False)
        return attribution_map.detach().cpu().numpy()
