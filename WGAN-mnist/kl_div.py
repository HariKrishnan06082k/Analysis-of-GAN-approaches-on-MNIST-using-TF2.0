import silence_tensorflow.auto
import tensorflow_probability as tfp

weather_A = tfp.distributions.Bernoulli(probs = 0.8)
weather_B = tfp.distributions.Bernoulli(probs = 0.7)

weather_A.sample(365)
print(tfp.distributions.kl_divergence(weather_A,weather_B))
