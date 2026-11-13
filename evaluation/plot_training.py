
import pandas as pd
import matplotlib.pyplot as plt

from training.config import LOG_DIR

log_path = LOG_DIR / "training_log.csv"

data = pd.read_csv(log_path)

if data.empty:
    raise RuntimeError("Training log is empty.")

# Plot training and validation loss.
plt.figure(figsize=(10, 5))

plt.plot(
    data["step"],
    data["train_loss"],
    marker="o",
    label="Training Loss",
)

plt.plot(
    data["step"],
    data["val_loss"],
    marker="o",
    label="Validation Loss",
)

plt.xlabel("Training Steps")
plt.ylabel("Cross Entropy Loss")
plt.title("MiniGPT Training Progress")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(LOG_DIR / "loss_graph.png", dpi=150)
plt.show()

# Plot learning rate.
plt.figure(figsize=(10, 5))

plt.plot(
    data["step"],
    data["learning_rate"],
    marker="o",
)

plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)

plt.tight_layout()
plt.savefig(LOG_DIR / "learning_rate_graph.png", dpi=150)
plt.show()

print("Graphs saved successfully.")
