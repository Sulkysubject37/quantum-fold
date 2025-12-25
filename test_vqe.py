from protein_folding_vqe import run_vqe_pipeline

def test_pipeline():
    sequence = "ACD"
    print(f"Running VQE for sequence: {sequence}")
    try:
        results = run_vqe_pipeline(sequence, num_layers=1)
        print("VQE run successful!")
        print(f"Optimal energy: {results['optimal_energy']}")
    except Exception as e:
        print(f"VQE run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
