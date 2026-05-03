import logging
import os
from src.tracker import WandbTracker
from src.cv import run_nested_cv
from src.sweep import run_sweep
from src.data import split_data, get_loader
from src.gnn import Model
from src.train import train_eval
from src.eval import validation


class GraphPipeline:
    def __init__(self, data, graph_type: str):
        self.data = data
        self.graph_type = graph_type
        self.tracker = WandbTracker()
        self._setup_logger()

    def _setup_logger(self):
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"pipeline_{self.graph_type}.log")

        self.logger = logging.getLogger(f"Pipeline_{self.graph_type}")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if the pipeline is re-instantiated
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def execute(self, audit_outer: int = 5, audit_inner: int = 2, sweep_count: int = 20, offline: bool = False):
        self.logger.info(f"STARTING PIPELINE FOR: {self.graph_type} (Offline: {offline})")

        # Step 3: Run Nested CV (Auditing)
        self.logger.info("Step 3: Running Nested CV (Auditing)...")
        try:
            generalization_auc, _ = run_nested_cv(
                self.data,
                self.graph_type,
                self.tracker,
                outer=audit_outer,
                inner=audit_inner,
                offline=offline,
            )
            self.logger.info(f"Audit complete. Generalization ROC AUC: {generalization_auc:.4f}")
        except Exception as e:
            self.logger.error(f"Error during Nested CV: {str(e)}")
            raise e

        # Step 4: Run Sweep (Optimization on 100% data)
        self.logger.info("Step 4: Running Hyperparameter Sweep (Optimization)...")
        try:
            sweep_results = run_sweep(self.data, self.graph_type, self.tracker, count=sweep_count, offline=offline)
            self.logger.info(f"Sweep Optimization complete: {sweep_results.get('status')}")
        except Exception as e:
            self.logger.error(f"Error during Sweep: {str(e)}")
            raise e

        self.logger.info(f"FINISHED PIPELINE FOR: {self.graph_type}")

        return {
            "graph_type": self.graph_type,
            "generalization_auc": generalization_auc,
            "sweep_status": sweep_results.get("status"),
        }

    def execute_dry_run(self, wandb_local: bool = False):
        """
        Runs a single training and validation cycle with default parameters.
        If wandb_local is True, it runs the full pipeline but in offline mode.
        """
        if wandb_local:
            return self.execute(audit_outer=3, audit_inner=2, sweep_count=5, offline=True)

        self.logger.info(f"STARTING DRY-RUN FOR: {self.graph_type}")

        # 1. Split Data
        self.logger.info("Splitting data (train 0.6, val 0.2, test 0.2)...")
        train_data, val_data, test_data = split_data(self.data, val_ratio=0.2, test_ratio=0.2)

        # 2. Create Loaders
        train_loader = get_loader(train_data, batch_size=128, shuffle=True)
        val_loader = get_loader(val_data, batch_size=128, shuffle=False)
        test_loader = get_loader(test_data, batch_size=128, shuffle=False)

        # 3. Initialize Model with default params
        self.logger.info("Initializing Model with default hyperparameters (hidden=64, out=32)...")
        model = Model(
            hidden_channels=64,
            out_channels=32,
            data=train_data,
        )

        # 4. Train
        self.logger.info("Starting training cycle (50 epochs)...")
        model, _ = train_eval(
            model,
            train_loader,
            val_loader,
            lr=0.005,
            show_progress=True,
            tracker=None,  # This disables W&B logging
        )

        # 5. Final Evaluation
        self.logger.info("Running final evaluation on test set...")
        test_loss, metrics = validation(model, test_loader)

        # Print results to console instead of logging for dry-run
        print("\n" + "=" * 50)
        print(f"DRY-RUN RESULTS FOR: {self.graph_type}")
        print("=" * 50)
        print(f"{'Metric':<20} | {'Value':<10}")
        print("-" * 35)
        print(f"{'Test Loss':<20} | {test_loss:.4f}")

        for key, value in metrics.items():
            if key == "class_report":
                continue
            if isinstance(value, (int, float)):
                print(f"{key:<20} | {value:.4f}")
            else:
                print(f"{key:<20} | {value}")

        if "class_report" in metrics:
            print("\nClassification Report:")
            print(metrics["class_report"])
        print("=" * 50 + "\n")

        self.logger.info(f"DRY-RUN COMPLETE for {self.graph_type}")

        return {
            "graph_type": self.graph_type,
            "test_loss": test_loss,
            "metrics": metrics,
        }
