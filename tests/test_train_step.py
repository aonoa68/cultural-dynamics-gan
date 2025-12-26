from placemotiongan.train import simulate_training

def test_simulate_training_step():
    logs, csv_path = simulate_training(total_steps=5)
    assert len(logs) == 5
