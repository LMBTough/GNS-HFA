# GNS-HFE
This repository contains the code for the paper "GNS-HFE: Enhancing Transferable Adversarial Attacks on Vision Transformers through Gradient Normalization Scaling and High-Frequency Adaptation"

To run the code, you need to install the following packages use environment.yml:
```
conda env create -f environment.yml
```

# Experiments

ViT models are all available in [timm](https://github.com/huggingface/pytorch-image-models) library. We consider four surrogate models (vit_base_patch16_224, pit_b_224, cait_s24_224, and visformer_small) and four additional target models (deit_base_distilled_patch16_224, levit_256, convit_base, tnt_s_patch16_224).

To evaluate CNN models, please download the converted pretrained models from ( https://github.com/ylhz/tf_to_pytorch_model) before running the code. Then place these model checkpoint files in `./models`.

#### Introduction


- `methods.py` : the implementation for GNS_HFE attack.

- `evaluate.py` : the code for evaluating generated adversarial examples on different ViT models.

- `evaluate_cnn.py` : the code for evaluating generated adversarial examples on different CNN models.
  

#### Example Usage

##### Generate adversarial examples:

- GNS_HFE

```
python attack.py --attack GNS_HFE --model_name vit_base_patch16_224 --scale --mhf search
```

You can also modify the hyper parameter values to align with the detailed setting in our paper.


##### Evaluate the attack success rate

- Evaluate on ViT models

```
bash run_evaluate.sh PATH_TO_ADVERSARIAL_EXAMPLES # PATH_TO_ADVERSARIAL_EXAMPLES is the path to the generated adversarial examples
```

- Evaluate on CNN models

```
python evaluate_cnn.py
```

## Acknowledgments

Code refer to: [Token Gradient Regularization](https://github.com/jpzhang1810/TGR) and [Towards Transferable Adversarial Attacks on Vision Transformers](https://github.com/zhipeng-wei/PNA-PatchOut) and [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model)