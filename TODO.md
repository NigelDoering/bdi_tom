# BDI-ToM Project TODO

## Experiment 2 (Goal Prediction)
- [x] Review earlier Experiment 2 workflow to confirm next implementation focus
- [x] Design data loader utilities to read distractor episodes and prepare batches per observation fraction
- [x] Implement evaluation script harness for Experiment 2 (model inference over fractions, metrics collection)
- [x] Run smoke test of evaluation script using existing checkpoints (or stub) and document output location
- [x] Draft visualization plan for Experiment 2 metrics
- [x] Implement plotting script to render distractor probability curves and peak comparison
- [x] Generate plots from current metrics and verify artifacts saved to exp_2 directory
- [x] Sanity check Experiment 2 evaluation: run baseline (full-trajectory) metrics and compare
- [x] Inspect POI index alignment between training mappings and evaluation mapping
- [ ] Log sample probabilities for true goal vs distractor across fractions

## V3 Model Improvements (SC-BDI-VAE)
- [x] Diagnose poor goal prediction accuracy in V2 model
- [x] Identify critical bug: `goal_idx` not passed to model in training
- [x] Create V3 model with:
    - [x] Conditional prior for intention: p(z_i | z_b, z_d)
    - [x] InfoNCE contrastive loss between z_d and goal
    - [x] Direct goal prediction from desire latent
    - [x] KL annealing schedule (monotonic warmup)
    - [x] Free-bits to prevent posterior collapse
- [x] Create V3 training script (`train_bdi_vae_v3.py`)
- [x] Document V2 vs V3 differences (`docs/V2_VS_V3_CRITICAL_ANALYSIS.md`)
- [ ] Run V3 training and compare to V2 baseline
- [ ] Validate that desire-to-goal accuracy improves
- [ ] Run ablation studies:
    - [ ] InfoNCE alone
    - [ ] Conditional prior alone
    - [ ] KL annealing alone
- [ ] Update methodology documentation (ICML_METHODOLOGY.tex) for V3

## Next Steps
- [ ] If V3 goal prediction is still poor, investigate embedding pipeline
- [ ] Consider counterfactual evaluation (trajectory perturbations)
- [ ] Explore sequential/temporal modeling (LSTM/Transformer over trajectory)
