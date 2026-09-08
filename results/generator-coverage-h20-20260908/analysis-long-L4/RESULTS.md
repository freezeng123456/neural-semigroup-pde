# H20 generator coverage results

Exploratory, best short-rollout validation checkpoint. Ratios below one favor A.

| Run / scheme / mode | A MSE | B MSE | A/B | Seed ratios | A/B grid source MSE | Accuracy gate |
|---|---:|---:|---:|---|---:|---|
| long-L4 / midpoint / detached | 2.0582e-06 | 2.18599e-06 | 0.9415 | 1.048, 0.902, 0.883 | 8.78502e-05 / 7.76156e-05 | fail |
| long-L4 / midpoint / initial | 1.95971e-06 | 2.65487e-06 | 0.7382 | 0.786, 0.748, 0.684 | 7.92169e-05 / 8.11851e-05 | pass (exploratory) |
| long-L4 / midpoint / teacher | 1.32579e-06 | 3.48177e-06 | 0.3808 | 0.111, 0.820, 0.606 | 6.55192e-05 / 5.20358e-05 | pass (exploratory) |
| long-L4 / midpoint / unroll | 1.8022e-06 | 1.69535e-06 | 1.0630 | 1.443, 0.869, 0.958 | 8.81966e-05 / 7.18466e-05 | fail |
