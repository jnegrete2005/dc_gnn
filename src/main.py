import argparse
import json
from copy import deepcopy

import torch
from src.pipeline import GraphPipeline


def main():
    parser = argparse.ArgumentParser(description="GNN Pipeline for Drug-Disease Link Prediction")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single training cycle without W&B logging",
    )
    parser.add_argument(
        "--wandb-dry-run",
        action="store_true",
        help="Run the nested cv with local-only W&B logging",
    )
    args = parser.parse_args()

    # --- Step 1: Define the graph to process ---
    # This matches the "Input" tracking section
    graph_path = "data/graph.pt"
    graph_type = "baseline_baseline_rich"

    print(f"Loading data for graph type: {graph_type}...")
    try:
        data = torch.load(graph_path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {graph_path} not found. Please ensure the graph is created first.")
        return

    # --- Step 2 & 3: Run the Modular Pipeline ---
    # This class handles the Logger, W&B Tracker, and the 5-step workflow orchestration
    pipeline = GraphPipeline(data, graph_type)

    if args.dry_run:
        results = pipeline.execute_dry_run()
    elif args.wandb_dry_run:
        results = pipeline.execute(audit_outer=3, audit_inner=2, offline=True)
    else:
        # Executes Step 3 (Nested CV Audit) and Step 4 (Bayesian Sweep Optimization)
        results = pipeline.execute(audit_outer=3, audit_inner=2, sweep_count=10)

    print(f"\nPipeline finished for {graph_type}.")

    if args.wandb_dry_run:
        print(f"Generalization ROC AUC (Nested CV): {results['generalization_auc']:.4f}")

        # Pretty print the detailed report for Nested CV stability
        if "report" in results:
            print("\n" + "=" * 60)
            print("NESTED CV DETAILED REPORT (STABILITY ANALYSIS)")
            print("=" * 60)

            printable_report = deepcopy(results["report"])
            for fold in printable_report:
                if "outer_metrics" in fold:
                    fold["outer_metrics"].pop("class_report", None)

            print(json.dumps(printable_report, indent=4))
            print("=" * 60 + "\n")

    elif not args.dry_run:
        print(f"Generalization ROC AUC (Nested CV): {results['generalization_auc']:.4f}")

    print(f"Detailed logs available at: log/pipeline_{graph_type}.log")


if __name__ == "__main__":
    main()
