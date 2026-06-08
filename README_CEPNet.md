# CEPNet Portable Package

This folder contains the code needed to migrate CEPNet independently from the original workspace.

## Files

- `cepnet.py`: standalone CEPNet model definition.
- `model/cepnet.py`: package-style CEPNet model used by training and validation scripts.
- `model/loss.py`: loss functions.
- `model/metrics.py`: evaluation metrics.
- `utils/Adan.py`: Adan optimizer.
- `utils/data.py`: SIRST and IRSTD-1k dataset loaders.
- `utils/lr_scheduler.py`: learning-rate scheduler.
- `train_cepnet.py`: training entry point.
- `val_cepnet.py`: validation entry point.

## Expected Dataset Layout

Run scripts from this folder or from a project root with the same relative dataset layout:

```text
datasets/
  SIRST/
    images/
    masks/
    trainval.txt
    test.txt
  IRSTD-1k/
    images/
    masks/
    trainval.txt
    test.txt
```

## Common Commands

```bash
python train_cepnet.py --dataset sirst --mode S
python val_cepnet.py --dataset sirst --mode S --checkpoint path/to/checkpoint.pth
```

