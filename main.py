import numpy as np
import matplotlib.pyplot as plt

# Time values
t_values = np.linspace(0, 10, 1000)

# Parameters
inertial_k1 = 0.3  # Inertial parameter (0 < k1 < 1)
intensity_k2 = 0.3  # Intensity parameter

# Normally distributed random variable ξ(t) with mean 0 and std deviation σ
mean = 0
sigma = 1
normal_distribution_xi_t = np.random.normal(mean, sigma, size=t_values.shape)

# Function f(t) = t^2
f_t = t_values**2

# Compute stimulus intensities
stimulus_intensity_I_t_1 = [intensity_k2 * (f_t[t] - f_t[t - 1]) for t in range(1, len(t_values))]
stimulus_intensity_I_t_2 = [
    (intensity_k2 * (f_t[t] - f_t[t - 1])) / f_t[t - 1] if f_t[t - 1] != 0 else 0
    for t in range(1, len(t_values))
]

# Initialize M_t
M_t = [0]

# Compute M_t iteratively
for t in range(1, len(t_values)):
    if f_t[t - 1] > 0:
        M_t.append(
            inertial_k1 * M_t[t - 1]
            + normal_distribution_xi_t[t]
            + stimulus_intensity_I_t_2[t - 1]
        )
    else:
        M_t.append(
            inertial_k1 * M_t[t - 1]
            + normal_distribution_xi_t[t]
            + stimulus_intensity_I_t_1[t - 1]
        )

# Plot the results
plt.plot(t_values, M_t, label="M(t)")
plt.xlabel("Time (t)")
plt.ylabel("M(t)")
plt.title("Dynamic Model Over Time")
plt.legend()
plt.grid()
plt.show()
