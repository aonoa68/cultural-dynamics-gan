# tests/test_scheduler.py
from placemotiongan.scheduler import make_scheduler

def test_lambda_reaches_one():
    sched = make_scheduler(total_steps=100, reach_portion=0.3, kind="linear")
    assert abs(sched(30) - 1.0) < 1e-6
