# TSEM
Implementation of TSEM: Temporally-Weighted Spatiotemporal Explainable Neural Network for Multivariate Time Series, in Pytorch 

## Install

```bash
$ pip install tsem
```

## Basic Usage

```python
import torch
from tsem.model import TSEM

series = torch.randn(1, 1, 256, 3)
window_size = series.shape[-2]
if window_size % 2 == 0:
    window_size -= 1

tsem = TSEM(
    window_size = window_size,
    time_length = series.shape[-2],
    feature_length = series.shape[-1],
    n_classes = 1000
)

preds = tsem(series) # (1, 1000)
```
## Parameters

- `window_size`: int.  
Window size along the time dimension. **Must be smaller than time_length and odd number**.
- `time_length`: int.  
Length of the series.
- `feature_length`: int.  
Number of series in the multivariate timeseries.
- `num_classes`: int.  
Number of classes to classify.

## Advanced Usage
- Training
```python
criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.Adam(model.parameters(), lr=1.5e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, datasets_size, 50)
```
- Explaining
```python
from tsem.feature_extractors.cams import XGradCAM
from tsem.utils.visualization import CAMFeatureMaps


extracting_model = model
extracting_module = model.cnn_layers1_b1
targeting_layer = "relu_12"

feature_maps = CAMFeatureMaps(XGradCAM)
feature_maps.load(extracting_model,extracting_module,targeting_layer,use_cuda=True)

mask1 = feature_maps.show(X_test[0], True, None, plot=True)
feature_maps.map_activation_to_input(mask1[0])
```
- Further details, please have a look in examples/
## Citation
```bibtex
@inproceedings{pham2023tsem,
  title={TSEM: Temporally-Weighted Spatiotemporal Explainable Neural Network for Multivariate Time Series},
  author={Pham, Anh-Duy and Kuestenmacher, Anastassia and Ploeger, Paul G},
  booktitle={Advances in Information and Communication: Proceedings of the 2023 Future of Information and Communication Conference (FICC), Volume 2},
  pages={183--204},
  year={2023},
  organization={Springer}
}
```
