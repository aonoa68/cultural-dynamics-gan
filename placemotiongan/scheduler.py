import numpy as np

class LambdaScheduler:
    """
    Lambda (exploration/exploitation weight) scheduler.

    mode:
      - "linear"
      - "logistic"
      - "delayed"
    """

    def __init__(self, total_steps: int, mode: str = "linear"):
        self.total_steps = int(total_steps)
        self.mode = mode

    def __call__(self, step: int) -> float:
        t = step / max(1, self.total_steps)
        if self.mode == "linear":
            return t
        elif self.mode == "logistic":
            k = 10.0
            x0 = 0.5
            return 1 / (1 + np.exp(-k * (t - x0)))
        elif self.mode == "delayed":
            if t < 0.3:
                return 0.0
            return (t - 0.3) / 0.7
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
