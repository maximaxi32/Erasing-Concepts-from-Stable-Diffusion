import torch
import matplotlib.pyplot as plt

# Define the range of values for the learning rate
start_lr = 1e-5
end_lr = 1.28e-2

# Define the number of steps in the schedule
num_steps = 1000

# Define the different schedules
schedules = {
    "linear": lambda step: start_lr + (end_lr - start_lr) * step / num_steps,
    "cosine": lambda step: end_lr + (start_lr - end_lr) / 2 * (1 + torch.cos(torch.tensor(step / num_steps * 3.1415))),
    "sigmoid": lambda step: (start_lr - end_lr) / (1 + torch.exp(torch.tensor((step - num_steps / 2) / (num_steps / 10)))) + end_lr,
    "squared": lambda step: start_lr + (end_lr - start_lr) * (step / num_steps) ** 2,
    "cube": lambda step: start_lr + (end_lr - start_lr) * (step / num_steps) ** 3,
    "exponential": lambda step: start_lr * (end_lr / start_lr) ** (step / num_steps)
}

# Plot the schedules
for name, func in schedules.items():
    lr_values = [func(step) for step in range(num_steps)]
    plt.plot(lr_values, label=name)

# Set the axis labels and a legend
plt.xlabel("Step")
plt.ylabel("Beta")
plt.yscale("log")
plt.legend()

# Display the plot
plt.show()
