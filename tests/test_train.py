from placemotiongan.train import simulate_training

def test_simulate_training_runs():
    logs, csv_path = simulate_training(total_steps=10)
    assert len(logs) == 10
