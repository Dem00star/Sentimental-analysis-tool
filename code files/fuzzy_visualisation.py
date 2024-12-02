import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Define the sentiment range (0 to 1)
sentiment_range = np.linspace(0, 1, 100)

# Define Gaussian membership functions for sentiment categories
positive = fuzz.gaussmf(sentiment_range, mean=0.85, sigma=0.12)
neutral = fuzz.gaussmf(sentiment_range, mean=0.5, sigma=0.15)
negative = fuzz.gaussmf(sentiment_range, mean=0.15, sigma=0.15)

# Plot the Gaussian membership functions
plt.figure(figsize=(10, 6))
plt.plot(sentiment_range, positive, label="Positive Sentiment", color="green")
plt.plot(sentiment_range, neutral, label="Neutral Sentiment", color="orange")
plt.plot(sentiment_range, negative, label="Negative Sentiment", color="red")

# Add labels, title, and legend
plt.title("Gaussian Membership Functions for Sentiments", fontsize=14)
plt.xlabel("Sentiment Score", fontsize=12)
plt.ylabel("Membership Value", fontsize=12)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True)

# Show the plot
plt.show()