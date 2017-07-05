import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

greyHeight = 28 + 4 * np.random.randn(greyhounds)
labHeight = 24 + 4 * np.random.randn(labs)

plt.hist([greyHeight, labHeight], stacked=True, color=['r','b'])
plt.show()

# this creates a histogram of randomly generated lab and greyhound heights based on a gaussian distribution
# we can see from this plot that height is a good feature for dogs that are at heights at the edge of this graph but
# becomes less useful as it approaches the middle
