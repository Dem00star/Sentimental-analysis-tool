import numpy as np
import matplotlib.pyplot as plt

# Define the universe of discourse
x = np.linspace(0, 10, 100)

# Define primary membership function
primary_membership = np.exp(-((x - 5) ** 2) / 4)

# Define upper and lower membership boundaries
upper_membership = np.minimum(primary_membership + 0.2, 1)
lower_membership = np.maximum(primary_membership - 0.2, 0)

# Plot the fuzzy type-2 set
plt.figure(figsize=(8, 5))
plt.fill_between(x, lower_membership, upper_membership, color='lightblue', label='Fuzzy Footprint of Uncertainty')
plt.plot(x, primary_membership, 'b-', label='Primary Membership Function')
plt.plot(x, upper_membership, 'r--', label='Upper Membership Boundary')
plt.plot(x, lower_membership, 'g--', label='Lower Membership Boundary')

# Add labels and legend
plt.title("Visualization of a Type-2 Fuzzy Set")
plt.xlabel("Universe of Discourse")
plt.ylabel("Membership Degree")
plt.legend(loc='best')
plt.grid(True)
plt.show()
