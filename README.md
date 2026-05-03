# MERIT-Net

MERIT-Net is a PyTorch project for image local tamper detection / forgery localization. It takes one RGB image as input and predicts a pixel-level tamper mask, a confidence map, and an image-level tamper score.

## Install

```bash
cd MERIT-Net
pip install -r requirements.txt
```

## Scan Data

Training data root is configured as `/data0/lzb-change-vmunet/FinalTrainData/`. The scanner recursively pairs images and masks, skips invalid samples, records `outputs/skipped_samples.csv` and `outputs/valid_samples.csv`, then creates balanced train/val splits:

```bash
python tools/scan_dataset.py --config configs/default_512.yaml
```

Split files are saved to:

```text
outputs/splits/train.txt
outputs/splits/val.txt
```

Each line is:

```text
image_path,mask_path,label
```

where `0` is authentic/negative and `1` is tampered/positive.

## Train

Single GPU:

```bash
python tools/train.py --config configs/default_512.yaml
```

Two-GPU DDP, for example on 2 NVIDIA 4090 cards:

```bash
bash tools/train_ddp.sh configs/default_512.yaml 2
```

Detached two-GPU DDP run with stdout/stderr redirected to a log file:

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup setsid /data0/hl/conda_envs/hldemo/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  -m tools.train \
  --config configs/default_512.yaml \
  > /home/hl/train_merit_net_s_512_stdout.log 2>&1 < /dev/null &
```

Follow logs:

```bash
tail -f /home/hl/train_merit_net_s_512_stdout.log
tail -f outputs/merit_net_s_512/logs/train.log
```

Training progress is updated in place on one stdout line per phase, with `Epoch current/total`, an ASCII progress bar, elapsed time, ETA, and loss. Epoch summaries include validation `pixel_f1`, `pixel_auc`, `image_auc`, IoU and FPR.

The default 512 config uses `batch_size_per_gpu: 16` and `accumulate_grad_batches: 1`, so two-GPU training has global batch size `16 x 2 x 1 = 32`.

Resume from latest checkpoint:

```bash
python tools/train.py --config configs/default_512.yaml --resume latest
```

Smoke-test a short run:

```bash
python tools/train.py --config configs/default_512.yaml --debug
```

## Checkpoints

Every epoch is saved independently under:

```text
outputs/merit_net_s_512/checkpoints/
```

Example:

```text
epoch1.pth
epoch2.pth
epoch3.pth
...
epoch80.pth
best_checkpoint.txt
latest_checkpoint.txt
```

This project intentionally does not save `best.pth`. To find the best checkpoint:

```bash
cat outputs/merit_net_s_512/checkpoints/best_checkpoint.txt
```

Then use the listed `best_checkpoint_path` for testing.

## Evaluate All Test Sets

```bash
python eval/eval_all_testsets.py --config configs/default_512.yaml --ckpt outputs/merit_net_s_512/checkpoints/epochXX.pth
```

Evaluation defaults to `--batch_size 64`. Lower it if test-time memory is insufficient.

The script evaluates:

```text
Casiav1, Columbia, NIST16, IMD2020, DSO-1, Korus
```

Outputs include `outputs/test_skipped_samples.csv`, per-dataset metric CSV files under `outputs/test_results/`, and `outputs/test_results_summary.csv`.

## Visualize

```bash
python tools/visualize_results.py --config configs/default_512.yaml --ckpt outputs/merit_net_s_512/checkpoints/epochXX.pth --input_dir your_images
```

Visualizations are saved to:

```text
outputs/visualizations/
```

## Main Ablation Switches

Edit the config fields below:

```yaml
model:
  use_residual_branch: true
  use_transformer_branch: true
  use_gated_fusion: true
  use_edge_loss: true
  use_confidence_head: true
  use_image_head: true
  use_family_head: false
  use_refinement: true
```

If both residual and transformer/global branches are disabled, model construction raises an error. Family head is disabled by default and will not be trained unless reliable family labels exist.

## Notes

- Input sizes `512`, `768`, and `896` are supported by setting `data.img_size` and `model.img_size`.
- Images are aspect-ratio preserved and zero-padded to the target size.
- `valid_region` masks padding, and losses/metrics ignore padded pixels.
- Validation and test transforms are deterministic; random augmentations are train-only.
- Progressive augmentation is configurable through `augmentation_schedule.enabled`.
