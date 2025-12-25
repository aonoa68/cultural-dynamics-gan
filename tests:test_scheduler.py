from placemotiongan.scheduler import LambdaScheduler

def test_linear_scheduler_increases():
    sched = LambdaScheduler(total_steps=10, mode="linear")
    vals = [sched(i) for i in range(11)]
    assert vals[0] <= vals[-1]
