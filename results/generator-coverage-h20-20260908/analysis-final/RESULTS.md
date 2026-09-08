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
| long-L4 / midpoint / detached | 2.0582e-06 | 2.18599e-06 | 0.9415 | 1.048, 0.902, 0.883 | 8.78502e-05 / 7.76156e-05 | fail |
| long-L4 / midpoint / initial | 1.95971e-06 | 2.65487e-06 | 0.7382 | 0.786, 0.748, 0.684 | 7.92169e-05 / 8.11851e-05 | pass (exploratory) |
| long-L4 / midpoint / teacher | 1.32579e-06 | 3.48177e-06 | 0.3808 | 0.111, 0.820, 0.606 | 6.55192e-05 / 5.20358e-05 | pass (exploratory) |
| long-L4 / midpoint / unroll | 1.8022e-06 | 1.69535e-06 | 1.0630 | 1.443, 0.869, 0.958 | 8.81966e-05 / 7.18466e-05 | fail |
| rate-L4 / midpoint / detached | 1.92824e-06 | 2.18109e-06 | 0.8841 | 0.815, 0.784, 1.082 | 8.98953e-05 / 7.53024e-05 | fail |
| rate-L4 / midpoint / initial | 1.65486e-06 | 1.88774e-06 | 0.8766 | 0.874, 0.907, 0.850 | 8.22064e-05 / 7.56924e-05 | pass (exploratory) |
| rate-L4 / midpoint / teacher | 1.5768e-06 | 2.53283e-06 | 0.6225 | 0.441, 0.775, 0.706 | 7.49487e-05 / 5.54906e-05 | pass (exploratory) |
| rate-L4 / midpoint / unroll | 1.83407e-06 | 2.39109e-06 | 0.7670 | 0.648, 0.821, 0.849 | 8.65596e-05 / 7.08665e-05 | pass (exploratory) |
| long-L8 / midpoint / detached | 1.17177e-06 | 1.74291e-06 | 0.6723 | 0.922, 0.533, 0.619 | 5.83687e-05 / 5.78914e-05 | pass (exploratory) |
| long-L8 / midpoint / initial_repeat | 2.19359e-06 | 5.71889e-06 | 0.3836 | 0.082, 0.869, 0.795 | 8.9217e-05 / 7.01313e-05 | pass (exploratory) |
| long-L8 / midpoint / teacher | 1.08861e-06 | 3.84594e-06 | 0.2831 | 0.358, 1.121, 0.057 | 5.61138e-05 / 4.01316e-05 | fail |
| long-L8 / midpoint / unroll | 1.16593e-06 | 1.5875e-06 | 0.7344 | 0.956, 0.642, 0.645 | 5.88912e-05 / 5.10272e-05 | pass (exploratory) |
