from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class Scheduler:
    def __init__(
        self,
        optimizer,
        num_training_steps,
        num_warmup_steps: int = None,
        scheduler_name: str = "get_linear_schedule_with_warmup",
    ):
        if num_warmup_steps is None:
            num_warmup_steps = 0.1 * num_training_steps
        if scheduler_name == "get_linear_schedule_with_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if scheduler_name == "get_cosine_with_hard_restarts_schedule_with_warmup":
            self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    def step(self):
        self.scheduler.step()
