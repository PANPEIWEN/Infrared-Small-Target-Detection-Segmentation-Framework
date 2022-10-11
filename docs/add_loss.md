## Add Loss Function

You need to follow the process below to add loss function.

### Add or rewrite loss function

You need to perform these operations
in [utils/loss.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/utils/loss.py).

_Notice: Do not repeat the sigmoid or softmax operation in the model output layer and loss function, and generally
perform this operation in loss function._

1. Add custom loss function

```python
class YourLossName(nn.Module):
    def __init__(self, args1, args2, ..., **kwargs):
        super(YourLossName, self).__init__()
        pass

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        loss = ...
        pass
        return loss
```

2. Add loss function in pytorch

```python
"""
We need to rewrite the loss function in pytorch here.
For example, we rewrite nn.BCEWithLogitsLoss.
"""
# 'BCEWithLogits' is the new class name, you can also use the original name 'BCEWithLogitsLoss'
class BCEWithLogits(nn.Module):
    # The parameters here need to be consistent with the parameters required by nn.BCEWithLogitsLoss, and must have **kwargs.
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, **kwargs):
        super(BCEWithLogits, self).__init__()
        # Pass the parameters in __init__ to nn.BCEWithLogitsLoss
        self.crit = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, pred, target):
        # Maybe a softmax or sigmoid operation is required.
        # If the data dimension is not correct, you need to perform the lifting and lowering operation here.
        # Calculate loss
        loss = self.crit(pred, target)
        return loss

```

3. Add the loss function class name to \_\_all__ in
   the [build/build_criterion.py](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/build/build_criterion.py)
   file. 

   _How to modify the config file to use loss function, please refer to [docs/add_model.md](https://github.com/PANPEIWEN/Infrared-Small-Target-Segmentation-Framework/blob/main/docs/add_model.md)._