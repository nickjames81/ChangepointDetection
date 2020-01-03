from __future__ import division
import matplotlib.pyplot as plt
import bayesian_changepoint_detection.generate_data as gd
import seaborn

partition, data = gd.generate_xuan_motivating_example(200,500)

import numpy as np
changes = np.cumsum(partition)

fig, ax = plt.subplots(figsize=[16, 4])
for p in changes:
  ax.plot([p,p],[np.min(data),np.max(data)],'r')
for d in range(2):
  ax.plot(data[:,d])

plt.show()


import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial

Q_full, P_full, Pcp_full = offcd.offline_changepoint_detection(data,partial(offcd.const_prior, l=(len(data)+1)),offcd.fullcov_obs_log_likelihood, truncate=-20)

fig, ax = plt.subplots(figsize=[18, 8])
ax = fig.add_subplot(2, 1, 1)
for p in changes:
  ax.plot([p,p],[np.min(data),np.max(data)],'r')
for d in range(2):
  ax.plot(data[:,d])
plt.legend(['Raw data with Original Changepoints'])
ax = fig.add_subplot(2, 1, 2, sharex=ax)
ax.plot(np.exp(Pcp_full).sum(0))
plt.legend(['Full Covariance Model'])
plt.show()