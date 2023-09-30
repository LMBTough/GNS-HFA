from captum.attr import Saliency

class SaliencyMap:
    def __init__(self, model):
        self.model = model
        self.saliencymap = Saliency(model)

    def __call__(self, data, target):
        return self.saliencymap.attribute(data, target).cpu().detach().numpy()