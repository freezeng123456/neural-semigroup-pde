# H20 generator coverage results

Exploratory, best short-rollout validation checkpoint. Ratios below one favor A.

| Run / scheme / mode | A MSE | B MSE | A/B | Seed ratios | A/B grid source MSE | Accuracy gate |
|---|---:|---:|---:|---|---:|---|
| wave1-euler / euler / detached | 0.00534948 | 0.00512201 | 1.0444 | 0.998, 1.021, 1.118 | 0.0232888 / 0.0224975 | fail |
| wave1-euler / euler / initial | 0.00690701 | 0.0064524 | 1.0705 | 1.114, 1.008, 1.093 | 0.0306594 / 0.0286325 | fail |
| wave1-euler / euler / initial_repeat | 0.00658956 | 0.00667188 | 0.9877 | 0.998, 0.873, 1.106 | 0.0287205 / 0.0296173 | fail |
| wave1-euler / euler / teacher | 0.00483198 | 0.0047387 | 1.0197 | 0.986, 0.966, 1.113 | 0.0186956 / 0.0189146 | fail |
| wave1-euler / euler / unroll | 0.00587726 | 0.00572233 | 1.0271 | 0.996, 0.979, 1.111 | 0.0258526 / 0.0254181 | fail |
| wave2-midpoint / midpoint / detached | 0.00494036 | 0.00482443 | 1.0240 | 0.996, 0.967, 1.116 | 0.0230169 / 0.0228354 | fail |
| wave2-midpoint / midpoint / initial | 0.00624861 | 0.00607435 | 1.0287 | 0.987, 1.016, 1.086 | 0.0295912 / 0.0288345 | fail |
| wave2-midpoint / midpoint / teacher | 0.00444107 | 0.00437648 | 1.0148 | 0.986, 0.956, 1.108 | 0.0190196 / 0.019278 | fail |
| wave2-midpoint / midpoint / unroll | 0.00534125 | 0.00521893 | 1.0234 | 0.993, 0.976, 1.106 | 0.025209 / 0.0248702 | fail |
| oracle / midpoint / oracle | 4.28725e-06 | 4.55255e-06 | 0.9417 | 0.859, 0.951, 1.023 | 2.11399e-05 / 2.05833e-05 | privileged diagnostic |
