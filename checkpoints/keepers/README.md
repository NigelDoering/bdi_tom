# Keepers - Important Model Checkpoints

This directory is for storing important model checkpoints that should be tracked in git.

## Purpose

While all checkpoints in `checkpoints/` are ignored by default, any models saved in this `keepers/` subdirectory will be committed to the repository.

## When to Save Here

Save models here when:
- ✅ You have a particularly good performing model you want to preserve
- ✅ You're ready to share a model with collaborators
- ✅ You need a model checkpoint for a paper/publication
- ✅ You want to ensure a specific model is backed up in version control

## Example

```bash
# After training, copy your best model here
cp checkpoints/baseline_transformer/best_model.pt keepers/baseline_v1_best.pt

# Or save directly during training
python models/training/train_baseline_transformer.py \
    --checkpoint_dir checkpoints/keepers/my_important_run
```

## ⚠️ Important Notes

- **File size**: Be mindful of model sizes. Git isn't ideal for very large files (>100MB)
- **Consider Git LFS**: For models >50MB, consider using Git Large File Storage
- **Naming convention**: Use descriptive names like `baseline_transformer_acc15.2_epoch45.pt`
- **Documentation**: Add notes here about what each saved model represents

## Saved Models

<!-- Document your important models here -->

### baseline_transformer_v1.pt
- Date: YYYY-MM-DD
- Performance: Top-1: X.X%, Top-5: Y.Y%
- Notes: Initial baseline model for paper

<!-- Add more models as you save them -->
