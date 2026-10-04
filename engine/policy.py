"""Run the trained policy with plain NumPy.

The policy is a tiny MLP (19 -> 64 -> 64 -> 2), so we don't need PyTorch
in production. export_policy() pulls the weights out of a Stable-Baselines3
PPO model once; NumpyPolicy loads them and does the same forward pass.
"""

import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        data = np.load(path)
        self.layers = []
        i = 0
        while f"w{i}" in data:
            self.layers.append((data[f"w{i}"], data[f"b{i}"]))
            i += 1
        self.out_w, self.out_b = data["out_w"], data["out_b"]

    def logits(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)   # SB3's PPO uses tanh between layers
        return x @ self.out_w.T + self.out_b

    def predict(self, obs, deterministic=True):
        """Same call shape as SB3's model.predict()."""
        return np.argmax(self.logits(obs), axis=-1), None


def export_policy(sb3_model, path):
    """Saves the actor network of a trained SB3 PPO model as an .npz file."""
    policy = sb3_model.policy
    linears = [m for m in policy.mlp_extractor.policy_net if hasattr(m, "weight")]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    arrays["out_w"] = policy.action_net.weight.detach().cpu().numpy()
    arrays["out_b"] = policy.action_net.bias.detach().cpu().numpy()
    np.savez(path, **arrays)
