# IRIS: Innovation Residual Iterative Stabilization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**IRIS** is an error-correcting optimizer that frames optimization as predictive error correction rather than gradient extrapolation. It tracks *innovation* - the prediction error between the current gradient and a smoothed estimate - and folds that error back into the update.

```python
from iris import IRIS

# Drop-in replacement for AdamW
optimizer = IRIS(
    model.parameters(), 
    lr=3e-3,
    betas=(0.96, 0.92, 0.9995),    # vision defaults
    weight_decay=0.01
)

for batch in dataloader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The formal writeup (math, proofs, related work) lives in a separate repo: [IRIS-documentation](https://github.com/madhavasthana-oss/iris-latex). This README covers usage and reports what has actually been tested.

---

## Status

IRIS is an active, personal research project - **not** a finished or peer-reviewed result. It's built and tested by one person on limited compute (currently free-tier / low-cost cloud GPUs). Here's what that means concretely:

- Validation so far is limited to **vision tasks** at small-to-moderate scale (CIFAR-10, CIFAR-100). No language modeling, diffusion, or RL experiments have been run yet. While I intend to further target this once compute becomes available, I will not pursue active testing until my studies are complete, and I can afford research utilities.
- Most results reported below come from **3 seeds**, not large statistical sweeps. 
- An earlier "heavy-ball" / split-error-correction variant was implemented and explored, then removed entirely from the codebase after it showed no measurable benefit over the standard formulation. It's mentioned here only so the history is transparent - it is not present in the current code.
- The accompanying paper is unsubmitted and unfinished ([IRIS-documentation](https://github.com/madhavasthana-oss/iris-latex)).

If you're evaluating this for research or hiring purposes: treat the results below as promising early signal from a small, honest experiment suite, not as a validated claim of general superiority over AdamW.

---

## Why IRIS?

### The problem with existing optimizers

**Adam/AdamW:**
- Treats all gradient variance the same way - no distinction between stochastic noise and genuine landscape difficulty
- Narrow practical learning-rate range (commonly capped around 3e-3)
- Typically needs extensive warmup (10-20 epochs) to train stably at scale

**AdaBelief:**
- Uses `(g_t - m_t)^2` as its "belief" term, which is a β1-scaled, biased version of true innovation, used in the denominator
- In our one CIFAR-100/ResNet-18 run (batch 2048), AdaBelief reached a lower training loss than IRIS but **2% lower test accuracy**, consistent with overfitting

**Adan:**
- Uses gradient difference `g_t - g_{t-1}`, which compares two noisy samples and has roughly 2× the variance of IRIS's innovation term
- In our toy-function benchmarks (Rosenbrock, Rastrigin) and CIFAR-100 run, Adan required extensive tuning and still diverged or oscillated; IRIS did not, in the same runs

### The IRIS approach

**Innovation-based error correction:**
```python
I_t     = g_t - g_hat_{t-1}              # Innovation (prediction error)
eps_t   = EMA[I_t]                       # Accumulated systematic error
g_hat_t = g_hat_{t-1} + psi1_inv * I_t   # Corrected gradient estimate
v_hat_t ~= (g_t + beta2 * I_t)^2         # Variance of the corrected gradient
```

**What this buys, based on testing so far:**
- Innovation tracks systematic bias rather than raw noise, which is why it holds up better at large batch size in the one setting I've tested (batch 2048, CIFAR-100)
- The correction term stays active near minima, instead of vanishing the way Adan's gradient difference does - shown on the Rosenbrock/Rastrigin toy landscapes
- In the CIFAR-100 run, IRIS reached lower test loss and higher test accuracy than AdamW - see [Results](#results) for the actual numbers

An earlier version of this README claimed IRIS trained ~25% faster in wall-clock time than AdamW. That number was an artifact of uneven logging overhead between runs, not a real speed advantage, and has been removed. With logging matched, IRIS was ~21 minutes slower than AdamW in the same run. I'm leaving this note here rather than quietly deleting the old claim, since it's exactly the kind of thing that should be traceable if you're deciding whether to trust the rest of this README.

---

## How it works

### 1. Innovation vs. gradient difference

**Adan's gradient difference** (compares two noisy samples):
```python
d_t = g_t - g_{t-1}
# Near minima, g_t ~= g_{t-1}, so d_t -> 0 and correction vanishes
```

**IRIS's innovation** (compares to a smooth estimate):
```python
I_t = g_t - g_hat_{t-1}
# Near minima, g_hat_{t-1} still lags, so I_t stays informative
# this becomes extremely important when using IRIS with high momentum, because overshoot, if not avoided, gets tamed otherwise.
```

### 2. Direct vs. inverse scaling

**AdaBelief** puts the innovation-like term in the denominator (inverse scaling):
```python
step ~= m_t / sqrt[(g_t - m_t)^2 + eps]
# Can divide by near-zero values when g_t ~= m_t
```

**IRIS** puts it in the numerator (direct scaling):
```python
step ~= (g_hat_t + beta2*eps_hat_t) / sqrt[v_hat_t + eps]
# Correction adjusts direction, not the scale of the step
```
My findings show, that the strategy of using **belief**, or rather, belief with **bias corrected momentum** called Innovation, as in Kalman Filtering theory, has 
advantages. 

AdaBelief and IRIS both use a gradient prediction-error signal, but with opposite responses: AdaBelief uses it to shrink steps (caution under surprise), IRIS uses it to correct future estimates (adaptation under surprise). In this run, both reached similar final test loss, but IRIS reached substantially higher test accuracy (74.02% vs. 72.5%), converging to 70% accuracy approximately 35 epochs earlier. I interpret this as the inverse-scaling in AdaBelief's denominator producing conservative updates that reach a lower-loss but less accurate region of weight space — consistent with the overfitting behavior reported in the original AdaBelief paper at large batch sizes. 
See [Results](#results).

### 3. Variance matching

```python
numerator   = g_hat_t + beta2 * eps_hat_t   # what we step with
denominator = sqrt[(g_t + beta2*I_t)^2]     # variance of what we step with
```

The denominator estimates uncertainty of the *corrected* gradient, not the raw one, so the preconditioner scale matches what's actually being applied.

### 4. Recursive bias correction (Kalman-inspired)

```python
# Standard Adam: needs an integer step count from the GPU
bias_correction = 1 - beta1**t

# IRIS: purely recursive, no GPU->CPU sync
psi1_t   = beta1 * psi1_{t-1} + 1
psi1_inv = 1 / psi1_t
```

This is algebraically a Kalman gain with static measurement noise. It removes the GPU-CPU synchronization Adam-style bias correction requires and gives exact bias correction from step 1. That does **not** mean warmup is unnecessary in general - our CIFAR-100 run happened to use none, but I haven't run the controlled comparison (same run, with vs. without warmup) needed to say whether IRIS benefits from one. Preliminary indications are that it can. Treat this as an open question, not a settled advantage.

### 5. Timescale separation

```python
beta1 = 0.98  ->  tracks the gradient over ~50 steps (slow)
beta2 = 0.92  ->  reacts to error over ~12 steps      (fast)
```

`beta2` needs to sit meaningfully below `beta1` (empirically, 0.15-0.25 lower) so error correction adapts faster than the error itself. If the two timescales are too close, the correction chases rather than catches.

---

## Installation

From source (no PyPI package yet):
```bash
git clone https://github.com/madhavasthana-oss/iris.git
cd iris
pip install -e .
```

---

## Quick start

```python
import torch
from iris import IRIS

model = YourModel()

optimizer = IRIS(
    model.parameters(),
    lr=3e-3,
    betas=(0.98, 0.92, 0.99),
    weight_decay=0.01,
    eps=1e-8
)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Optional SNR-based clipping

```python
optimizer = IRIS(
    model.parameters(),
    lr=3e-3,
    betas=(0.98, 0.92, 0.99),
    snr_threshold=4.0,   # enables trust-region clipping
    weight_decay=0.01
)
```

---

## Hyperparameter guide

### Learning rate

I have only tuned and tested learning rate in the **vision** settings listed under [Results](#results) (CIFAR-10/100, batch sizes up to 2048). For those settings, `3e-3` to `1e-2` has worked well. I have **not** tested IRIS on language models, diffusion models, or fine-tuning, so no LR guidance is given for those regimes - if you try it there, I'd genuinely like to know what you find.

### Betas

```python
betas = (beta1, beta2, beta3)
```
Default: `(0.98, 0.92, 0.99)`, used in all reported results below.

- **beta1** - gradient estimate EMA (typically 0.9-0.99). Higher = smoother, slower adaptation.
- **beta2** - innovation residual EMA (typically 0.8-0.95). Must sit 0.15-0.25 below beta1 for timescale separation; lower = faster correction.
- **beta3** - variance EMA (typically 0.99-0.9995). Higher = more stable variance estimates.

### SNR threshold (`snr_threshold`)

Controls optional trust-region clipping:
```python
denominator = rho * sqrt(v_hat_t) + eps
```

| Value | Effect |
|-------|--------|
| `None` (default) | No clipping |
| `4.0` | Conservative, used if training is unstable |
| `2.0` | More aggressive clipping |

I have not systematically swept this parameter - the default `None` is what all reported results use unless stated otherwise.

### Weight decay

Decoupled, AdamW-style:
```python
theta_t = (1 - eta*lambda) * theta_{t-1} - eta * update
```
Typical values: 0.01-0.1. Not swept beyond the default `0.01` used in reported results.

---

## Results

Everything below is what has actually been run, with seed counts stated. This section will be updated as more experiments complete - see [Status](#status) for what's still missing.

### CIFAR-100, ResNet-18, batch 2048, 200 epochs (1 run)

| Optimizer   | Warmup used | Final Test Acc | Best Test Acc     | 73% first crossed | Train Time |
|-------------|-------------|----------------|-------------------|-------------------|------------|
| AdamW       | 10 epochs   | 73.55%         | 73.63% (ep 178)   | Epoch 108         | ~150 min   |
| AdaBelief   | 10 epochs   | 72.58%         | 72.67% (ep 189)   | Never             | ~150 min   |
| Adan        | 5–20 epochs | <73%, diverged | —                 | —                 | —          |
| **IRIS**    | 10 epochs   | **74.02%**     | **74.12% (ep 152)** | **Epoch 89**    | ~170 min   |

Across 3 runs per optimizer, not yet repeated across seeds - treat as early signal, not a confirmed effect size. IRIS reached higher accuracy in this run, but this compares "IRIS, no warmup" against "AdamW, 20-epoch warmup," not a controlled ablation - I haven't yet tested IRIS with warmup, or AdamW without it, in the same run.

**On training time:** an earlier version of this table reported IRIS as ~25% faster (113 min vs. 150 min). That gap turned out to be a logging-overhead artifact, not a real difference in optimizer speed. With logging matched, IRIS was **~21 minutes slower** than AdamW in this run, while reaching ~0.8 points higher test accuracy. I'm keeping this correction visible rather than just fixing the number silently.

### CIFAR (ResNet family), 2 seeds, AdamW vs. IRIS

Same architecture and hyperparameters across both optimizers, run at seeds `42` and `3407`. IRIS reached a lower final test loss than AdamW on both seeds (~1.68-1.70 vs. ~1.87-1.89), with the gap consistent in direction across seeds. Test accuracy showed a similar, smaller edge for IRIS. Raw WandB logs available on request / linked in the docs repo.

This is 2 seeds, not a large sweep - enough to say the direction is consistent, not enough to claim a precise effect size.

### Toy optimization landscapes

On Rosenbrock and Rastrigin, IRIS tolerated learning rates roughly 10-1000× higher than Adam-family optimizers before diverging, and was the only optimizer tested that reached the global minimum on Rastrigin. Full numbers are in the paper (IRIS-documentation repo).

### In progress

- **ViT-Small, CIFAR-100** - currently running, no results yet.
- **CNN, CIFAR-10** - currently running, no results yet.
- **Large-batch stress test** (pushing batch size well past 2048 to test the core claim directly) - planned.

### Not yet attempted

- Language modeling (any scale)
- Diffusion models
- Anything beyond CIFAR-scale vision (e.g. ImageNet or subsets like Tiny-ImageNet)

---

## Algorithm

```python
# Bias correction accumulators
psi1_t = beta1*psi1_{t-1} + 1
psi2_t = beta2*psi2_{t-1} + 1
psi3_t = beta3*psi3_{t-1} + 1

# Innovation
I_t = g_t - g_hat_{t-1}

# Gradient estimate
g_hat_t = g_hat_{t-1} + psi1_inv * I_t

# Innovation residual (error accumulation)
eps_hat_t = eps_hat_{t-1} + psi2_inv * (I_t - eps_hat_{t-1})

# Variance of corrected gradient
Sigma_t = (g_t + beta2*I_t)^2 - v_hat_{t-1}
v_hat_t = v_hat_{t-1} + psi3_inv * Sigma_t

# Parameter update
numerator   = g_hat_t + beta2*eps_hat_t
denominator = sqrt(v_hat_t) + eps
theta_t = (1 - eta*lambda)*theta_{t-1} - eta*(numerator / denominator)
```

With optional SNR clipping:
```python
update = clip(numerator / (rho*sqrt(v_hat_t) + eps), -1, 1)
theta_t = (1 - eta*lambda)*theta_{t-1} - eta*update
```

This matches the current implementation in `functional.py` exactly - the earlier split-error-correction / heavy-ball variant described in older versions of this repo has been fully removed from the code and is not reflected above.

---

## Comparison with other optimizers

| Property | Adam | AdaBelief | Adan | IRIS |
|----------|------|-----------|------|------|
| Core quantity | `g_t` | `g_t - m_t` | `g_t - g_{t-1}` | `g_t - g_hat_{t-1}` |
| Interpretation | Raw gradient | Biased innovation | Gradient velocity | Innovation |
| Near-minima behavior | - | Overfit (observed, 1 run) | Correction vanishes | Stays active (observed, toy landscapes) |
| Memory buffers | 2 | 2 | 4 | 3 |
| Warmup used in our CIFAR-100 run | Yes (20 ep) | Yes (20 ep) | Yes (10-20 ep) | None - untested whether IRIS needs it in general |
| Batch 2048, CIFAR-100 (1 run) | 73.0% acc | 71.5% acc | Diverged | 73.8% acc, ~21 min slower than AdamW |

---

## Advanced usage

### Mixed precision

```python
from torch.cuda.amp import autocast, GradScaler

optimizer = IRIS(model.parameters(), lr=3e-3)
scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Learning rate schedules

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = IRIS(model.parameters(), lr=3e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

Our one CIFAR-100 run didn't use warmup, but I haven't run the controlled comparison to know whether IRIS generally benefits from warmup or schedules - preliminary indications are that it might. Don't take the absence of warmup in that run as a recommendation against using one.

### AMSGrad variant

```python
optimizer = IRIS(
    model.parameters(),
    lr=3e-3,
    betas=(0.98, 0.92, 0.99),
    amsgrad=True
)
```

Implemented and available; not separately benchmarked yet.

---

## FAQ

**Is IRIS a drop-in replacement for Adam?**
Mechanically yes - same interface. Whether it's a strict improvement for your task is untested outside the settings in [Results](#results).

**Do I need warmup?**
Unclear. Our one CIFAR-100 run didn't use it and still trained stably, but I haven't run the controlled ablation, and early indications suggest IRIS can benefit from warmup in at least some settings. Try it both ways for your task rather than assuming it's unnecessary.

**Why does beta2 need to be lower than beta1?**
Error correction has to adapt faster than the error itself, or the correction always lags. This is a design constraint from the derivation, not an empirical tuning finding.

**What's the difference from AdaBelief?**
AdaBelief's `g_t - m_t` uses the *current* momentum (a look-ahead) and puts it in the denominator. IRIS's `g_t - g_hat_{t-1}` uses only the *previous* estimate (causal) and puts it in the numerator.

**What's the difference from Adan?**
Adan compares two noisy gradient samples directly, which has higher variance and vanishes near minima. IRIS compares to a smoothed estimate instead.

**Can I use gradient clipping alongside `snr_threshold`?**
You can, but they overlap in function - using both may over-constrain updates. Not something I've tested combinations of.

**Has this been tested outside vision?**
No. See [Status](#status).

---

## Contributing

This is a small, early-stage project and outside input is genuinely welcome, especially:
- Runs on tasks outside vision (NLP, RL, generative models)
- Independent reproduction of the CIFAR-100 / toy-landscape results
- Ablations (beta2/beta1 ratio, variance formulation)
- Convergence/regret-bound analysis beyond what's in the current draft paper

---

## License

MIT - see [LICENSE](LICENSE).

---

## Acknowledgments

- **Adam/AdamW** - the baseline this work is measured against throughout
- **AdaBelief** - motivated thinking carefully about what "prediction error" should mean in an optimizer
- **Adan** - motivated moving beyond raw gradient differences
- **Kalman filter theory** - the recursive-estimation framing behind the bias-correction scheme