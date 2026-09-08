# H20 generator coverage results

Exploratory, best short-rollout validation checkpoint. Ratios below one favor A.

| Run / scheme / mode | A MSE | B MSE | A/B | Seed ratios | A/B grid source MSE | Accuracy gate |
|---|---:|---:|---:|---|---:|---|
| rate-L4 / midpoint / detached | 1.92824e-06 | 2.18109e-06 | 0.8841 | 0.815, 0.784, 1.082 | 8.98953e-05 / 7.53024e-05 | fail |
| rate-L4 / midpoint / initial | 1.65486e-06 | 1.88774e-06 | 0.8766 | 0.874, 0.907, 0.850 | 8.22064e-05 / 7.56924e-05 | pass (exploratory) |
| rate-L4 / midpoint / teacher | 1.5768e-06 | 2.53283e-06 | 0.6225 | 0.441, 0.775, 0.706 | 7.49487e-05 / 5.54906e-05 | pass (exploratory) |
| rate-L4 / midpoint / unroll | 1.83407e-06 | 2.39109e-06 | 0.7670 | 0.648, 0.821, 0.849 | 8.65596e-05 / 7.08665e-05 | pass (exploratory) |
