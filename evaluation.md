# Evaluation of the model from Nested Cross-Validation

In this file, I am going to report the results of the evaluation of the model
from Nested Cross-Validation (CV), in order to keep track of the performance of
the model and to compare it with other models in the future.

As of right now, March 26, I have ran the nested CV with the following parameter
grid:

```python
param_grid = {
    "lr": [0.005, 0.001],
    "hidden_channels": [64, 128],
    "out_channels": [32, 64],
}
```

with 5 outer and 2 inner folds.

## Baseline graph

This models have the following architecture:

- Features of dimension 512.
- Preamble and postamble Fully-Connected NN of 3 layers with ReLU activation.
- Two SageConv layers with ReLU activation.
- No dropout or regularization.
- A dot product decoder.

Using the graph with ones as features, I got the following results:

```bash
Outer Fold 1: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 128, 'out_channels': 32}
Outer Fold 2: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 64, 'out_channels': 64}
Outer Fold 3: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 64, 'out_channels': 32}
Outer Fold 4: Best Hyperparameters: {'lr': 0.001, 'hidden_channels': 128, 'out_channels': 64}
Outer Fold 5: Best Hyperparameters: {'lr': 0.001, 'hidden_channels': 64, 'out_channels': 64}
```

Using the graph with rich features, I got the following results:

```bash
Final Generalization Loss from Nested CV: 0.4598
Outer Fold 1: Best Hyperparameters: {'lr': 0.001, 'hidden_channels': 128, 'out_channels': 64}
Outer Fold 2: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 64, 'out_channels': 64}
Outer Fold 3: Best Hyperparameters: {'lr': 0.001, 'hidden_channels': 64, 'out_channels': 32}
Outer Fold 4: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 64, 'out_channels': 64}
Outer Fold 5: Best Hyperparameters: {'lr': 0.005, 'hidden_channels': 64, 'out_channels': 32}
```

We can see that the model greatly benefits from the rich features, as the
generalization loss is significantly lower than the one from the baseline graph.

Based on these results, I conclude that this basic model should have:

- `lr`: 0.005
- `hidden_channels`: 64
- `out_channels`: 64
