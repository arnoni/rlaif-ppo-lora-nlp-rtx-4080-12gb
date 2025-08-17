# src/rlaif_pipeline/cli.py
import argparse
from .pipeline import main, RLAIFConfig, ModelConfig, LoRAConfig, TrainingConfig, DataConfig

def train_main():
    """Entry point for the rlaif-train command."""
    parser = argparse.ArgumentParser(description="Run the RLAIF training pipeline V4.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help="Overrides the objective in the config file."
    )
    
    args = parser.parse_args()
    
    if args.config:
        config = RLAIFConfig.load(args.config)
        # Command-line argument for objective takes precedence
        if args.objective:
            config.training.objective = args.objective
    else:
        # Create a default config if no file is provided
        default_objective = args.objective if args.objective else "harmless"
        config = RLAIFConfig(
            model=ModelConfig(),
            lora=LoRAConfig(),
            training=TrainingConfig(objective=default_objective),
            data=DataConfig()
        )
    
    main(config)

# You can add functions for eval_main and config_main here later
def eval_main():
    print("Evaluation script not implemented yet.")

def config_main():
    print("Config creation helper not implemented yet.")