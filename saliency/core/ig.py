from captum.attr import IntegratedGradients

class IntegratedGradient:
    def __init__(self, model):
        self.model = model
        self.saliency = IntegratedGradients(model)

    def __call__(self, data, target, gradient_steps=50):
        attribution_map = self.saliency.attribute(data,
                                                  target=target.squeeze(),
                                                  baselines=None,
                                                  n_steps=gradient_steps,
                                                  method="riemann_trapezoid")
        return attribution_map.detach().cpu().numpy()