# Final Model Data Flow and Module Explanation

This document is meant to help manually redraw a clean deep-learning-paper-style diagram.

## Overall pipeline

Input: one degraded LoViF image `x`.

Output: one restored RGB JPEG with the same file name and spatial size.

The final checkpoint contains two modules:
1. DiffUIR clean-310 backbone
2. MoCE-Prompt residual refiner v2-150k

The inference script loads one single `.pt` file and runs both modules sequentially.

## Module 1: DiffUIR clean-310 backbone

Code:
- `source/src/model.py`
- `source/infer_single_ckpt_jpeg96.py`

Role:
- Produces the first-stage restoration `y0`.
- Initialized from DiffUIR `model-300.pt`, then fine-tuned with official DiffUIR loss to `model-310.pt`.

Draw as:
- Large green block: `DiffUIR clean-310 backbone`
- Input arrow: `LQ image x`
- Output arrow: `intermediate restoration y0`

## Module 2: 9-channel refiner input

Code logic:
- `z = concat(x, y0, y0 - x)`

Role:
- `x`: original degraded image
- `y0`: clean-310 restored image
- `y0-x`: residual cue showing what the backbone changed

Draw as:
- Yellow concat block: `Concat [x, y0, y0-x] -> 9 channels`

## Module 3: Feature stem

Code:
- `source/models/moce_prompt_refiner.py`

Role:
- Converts the 9-channel input into width-32 features with a shallow convolution.

Draw as:
- `3x3 Conv Stem, C=32`

## Module 4: MoCE-Prompt trunk

Code:
- `source/models/moce_prompt_refiner.py`

Role:
- Four lightweight restoration blocks.
- Each block has shared processing and expert-style paths.
- It gives implicit degradation awareness without explicit task classification.

Draw as:
- Purple block: `MoCE-Prompt Blocks x4`
- Internal sub-boxes: `identity/protect`, `detail`, `weather correction`, `illumination/color`.

## Module 5: Prompt/gating branch

Code:
- `source/models/moce_prompt_refiner.py`

Role:
- Global pooled features predict sample-adaptive expert weights.
- It is not a degradation classifier.

Draw as:
- Side branch: `Global Pooling -> Gating MLP -> expert weights`, arrow back to MoCE trunk.

## Module 6: Residual and mask heads

Code:
- `source/models/moce_prompt_refiner.py`

Role:
- Residual head predicts correction `r`.
- Mask head predicts correction mask `m`.

Draw as:
- Two orange heads: `Residual head r` and `Mask head m`.

## Module 7: Conservative final correction

Formula:
- `y = clip(y0 + alpha * m * r, 0, 1)`
- `alpha = 0.15`

Role:
- Avoids over-editing.
- Protects clean-310 blur/detail restoration.
- Allows local correction of low-light/weather residuals.

Draw as:
- Merge node: `y0 + 0.15 * mask * residual`
- Output: `final restored image y`

## Module 8: JPEG96 saving

Code:
- `source/infer_single_ckpt_jpeg96.py`

Role:
- Saves RGB JPEG with `quality=96, subsampling=0, optimize=True`.
- Keeps input file name and input spatial size.

Draw as:
- Output block: `Save JPEG q=96, same name, same size`.
