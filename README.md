# AIC2026 — Tri-Modal Object Detection (RGB + Infrared + Depth)

Competition solution for **AIC2026**, a tri-modal object-detection task: each image provides
three aligned modalities — **RGB**, **Infrared** and **Depth** — and the model must predict
boxes for **12 classes**. Built on **[D-FINE](https://github.com/Peterande/D-FINE)**.

This repository contains **code, configs, and evidence logs only**. No competition data,
no checkpoints and no submission archives are included (see [What is *not* here](#what-is-not-here)).

---

## Results

| Item | Value |
|---|---|
| Best public-leaderboard submission | **50.0** (val 0.5670, scales `800+928`) |
| Improved submission (delivered) | val **0.5752** (+0.82 pp), scales `800+1024+1088`, APs 0.3053 → 0.3365 |
| Improvement cost | **zero retraining** — same checkpoint, different multi-scale TTA schedule |
| Halving replication | **19 / 20** random splits favour the new schedule (paired) |

Claim discipline (kept deliberately conservative):

* The scale combination was **selected on the same 300-image val set** that reports it, so the
  +0.82 pp is ≈ 0.5σ at the val standard error (±1.6 pp). **It is not claimed to be statistically
  significant**, and is expected to regress toward zero on an unseen test set.
* Halving does **not** remove the winner's curse (both halves are subsets of the selection set);
  it only shows the gain is stable *on this data*.
* Images within one sequence are strongly correlated (seen sequences 0.5509 vs unseen 0.6345),
  so a cluster bootstrap over sequences is the appropriate uncertainty estimate and is listed as
  the next step.

---

## Findings worth reading

**1. A gap in the multi-scale sweep was worth +0.82 pp.** The team had tested
704/800/832/864/896/928/992/1280 but **skipped 1024** (jumping straight to 1280, which collapses
to 0.5288). Filling that gap gives the best schedule. Raw evidence: [`evidence/ms1024.log`](evidence/ms1024.log),
[`ms1024b.log`](evidence/ms1024b.log), [`ms_hires.log`](evidence/ms_hires.log).

**2. Tri-modal grafting did essentially nothing.** A residual `ZeroConv` fusion branch for the IR
and depth channels was implemented and ablated on the same weights and the same 300 images:

| Input | mAP |
|---|---|
| full 5 channels | 0.5678 |
| infrared zeroed | 0.5678 |
| depth zeroed | 0.5678 |

Under the same protocol the plain RGB baseline is 0.5670, i.e. **the tri-modal graft is worth
+0.0008 — inside the noise**. Importantly, the ablation only proves that *the residual signal was
not effectively injected*; it does **not** prove the modalities are useless. See
`docs/02-architecture-v2.md` for the follow-up analysis and the depth-encoding hypothesis
(`/80` linear scaling with a 20.4 m clip may collapse the long tail).

**3. Two format traps that silently break inference.**

| | count | format | correct handling |
|---|---|---|---|
| depth `.png` | 883 | `I;16` uint16, 0–19999 mm | **must divide by 80** |
| depth `.jpg` | 117 | RGB uint8, 0–255 (already quantised) | use as-is |
| infrared | 1000 | 3 identical channels | take channel 0 |

**4. The public/private gap is real generalisation loss, not a preprocessing bug.** Per-channel
statistics of train / val / public-test / private-test are all within ~2 % of each other, ruling out
channel-order or gamma mistakes.

**5. Submission compliance is a hard gate.** `tools/submission/fix_submit_any.py` clamps boxes to
the image, then validates: exact file count, no nesting, no extra/missing files, ≤ 100 boxes per
image, no degenerate boxes, class set within range. Full checker output is in
[`docs/01-handoff.md`](docs/01-handoff.md).

---

## Repository layout

```
configs/                    30 D-FINE training / tri-modal configuration files
tools/
  inference/                multi-scale TTA, single-scale, tri-modal inference, box voting
  evaluation/               val sweeps, ablations, halving replication, cluster/strata analysis
  training/                 tri-modal init generation, EMA init, config generation, checkpoint checks
  audit/                    dataset/leak audit, class-mapping and detection sanity checks
  submission/               clamp + compliance validation, packaging
scripts/                    59 launcher scripts (go_*.sh = individual experiment runs,
                            run_*.sh = main entry points)
docs/
  01-handoff.md             complete engineering handoff, all numbers and file lists
  02-architecture-v2.md     tri-modal architecture upgrade plan
  03-submission-v2.md       finals submission notes
  04-conclusions-and-next-steps.md
evidence/                   raw logs backing the results above (small, text)
```

### Key entry points

| Task | Script |
|---|---|
| Multi-scale TTA evaluation (the +0.82 pp result) | `tools/evaluation/ms_tta.py` |
| Halving replication / paired comparison | `tools/evaluation/eval_split.py` |
| Channel ablation | `tools/evaluation/eval_ablate.py` |
| Tri-modal inference | `tools/inference/infer_tri.py` |
| Clamp + compliance check | `tools/submission/fix_submit_any.py` |
| Training launcher | `scripts/run_train.sh` |

---

## Reproduction

**Environment**

```bash
pip install -r requirements.txt
# D-FINE itself is not vendored here — clone it and its dependencies:
git clone https://github.com/Peterande/D-FINE
```

**Data.** Obtain the AIC2026 tri-modal dataset from the organisers and lay it out as the configs
expect (`dataset/` with `train/`, `val/`, `test/`). The dataset is **not redistributable** and is
therefore not included.

**Weights.** Replace the placeholder paths in the configs with your own D-FINE checkpoints
(`dfine_l_obj365.pth` etc.). The released D-FINE weights come from the official repo.

**Note on absolute paths.** The scripts were developed on a shared lab server and contain hard-coded
paths of the form `/home/<user>/...`. They are kept as-is so that the run-to-result mapping remains
auditable; retarget them to your own layout before running.

```bash
# evaluate one TTA schedule
python tools/evaluation/ms_tta.py --size 800 1024 1088 --nms-iou 0.7 --max-det 100

# validate a submission directory before upload
python tools/submission/fix_submit_any.py --dir submit/
```

---

## What is *not* here

| Excluded | Size | Why |
|---|---|---|
| `dataset/` | ~17 GB | competition data, not redistributable |
| `output/` | ~6.8 GB | training checkpoints |
| `weights/` | ~488 MB | third-party D-FINE pretrained weights |
| `submit/`, `zip/` | ~236 MB | submission archives |

`.gitignore` excludes all of the above, so a fresh clone will not attempt to fetch them.

---

## Licence and attribution

Code released under the **MIT Licence** (see `LICENSE`).
Built on **D-FINE**; see `docs/` for the specific techniques borrowed and the corresponding results.
The competition dataset and organiser materials remain the property of their respective owners.
