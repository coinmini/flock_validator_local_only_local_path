"""
Local-only validation entrypoint.
Runs model inference + LLM evaluation without any Flock API calls.

Usage:
    uv run python local_validate.py \
        --model-path /mnt/smh/swift_converted \
        --validation-file /opt/flock/task21/task21_eval_final.jsonl \
        --base-model-path /mnt/smh/huggingface_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd0a36b00ddf55e13ccb7b8cd0a \
        --is-lora \
        --eval-with-llm

Requires OPENAI_API_KEY and OPENAI_BASE_URL environment variables when --eval-with-llm is used.
"""

import sys
import click
from loguru import logger
from validator.modules.llm_judge import LLMJudgeValidationModule, LLMJudgeConfig, LLMJudgeInputData


@click.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the model weights (LoRA adapter directory or full model directory)",
)
@click.option(
    "--validation-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to local JSONL validation dataset",
)
@click.option(
    "--base-model-path",
    default=None,
    type=click.Path(exists=True),
    help="Local path to the base model snapshot (skips HuggingFace download)",
)
@click.option(
    "--is-lora",
    is_flag=True,
    default=False,
    help="Whether the model is a LoRA adapter",
)
@click.option(
    "--eval-with-llm",
    is_flag=True,
    default=False,
    help="Use LLM judge for evaluation (requires OPENAI_API_KEY and OPENAI_BASE_URL env vars)",
)
@click.option(
    "--base-model-name",
    default=None,
    type=str,
    help="Base model name for generation config (e.g. 'Qwen/Qwen3.5-4B'). "
         "Auto-detected from adapter_config.json when --is-lora is set.",
)
@click.option(
    "--context-length",
    default=2048,
    type=int,
    show_default=True,
    help="Maximum context length for generation",
)
@click.option(
    "--max-params",
    default=None,
    type=int,
    help="Maximum allowed model parameters (skip check if not set)",
)
@click.option(
    "--gen-batch-size",
    default=1,
    type=int,
    show_default=True,
    help="Batch size for generation",
)
@click.option(
    "--eval-batch-size",
    default=16,
    type=int,
    show_default=True,
    help="Batch size (number of parallel workers) for LLM evaluation",
)
def main(
    model_path: str,
    validation_file: str,
    base_model_path: str | None,
    is_lora: bool,
    eval_with_llm: bool,
    base_model_name: str | None,
    context_length: int,
    max_params: int | None,
    gen_batch_size: int,
    eval_batch_size: int,
):
    """Run local-only model validation (no Flock API needed)."""

    # Resolve base_model_name
    if base_model_name is None and is_lora:
        import json
        from pathlib import Path

        adapter_config_path = Path(model_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            logger.error(f"adapter_config.json not found in {model_path}")
            sys.exit(1)
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        logger.info(f"Auto-detected base model from adapter_config.json: {base_model_name}")

    if base_model_name is None:
        logger.error(
            "Cannot determine base model name. "
            "Use --base-model-name or --is-lora with an adapter_config.json."
        )
        sys.exit(1)

    config = LLMJudgeConfig(
        gen_batch_size=gen_batch_size,
        eval_batch_size=eval_batch_size,
    )

    logger.info("Initializing LLM Judge module...")
    module = LLMJudgeValidationModule(config=config)

    # Build eval_args
    eval_args = {
        "eval_require": 3,
        "gen_require": 1,
    }
    if not eval_with_llm:
        # When not using LLM eval, set eval_require to 0 to skip evaluation stage
        eval_args["eval_require"] = 0

    # Construct input data — validation_set_url and hg_repo_id are unused in local mode
    input_data = LLMJudgeInputData(
        hg_repo_id="local",
        revision="local",
        context_length=context_length,
        max_params=max_params or 0,
        validation_set_url="local",
        base_model=base_model_name,
        eval_args=eval_args,
    )

    logger.info(f"Model path:       {model_path}")
    logger.info(f"Validation file:  {validation_file}")
    logger.info(f"Base model:       {base_model_name}")
    logger.info(f"Base model path:  {base_model_path or '(download from HuggingFace)'}")
    logger.info(f"Is LoRA:          {is_lora}")
    logger.info(f"Eval with LLM:    {eval_with_llm}")
    logger.info(f"Context length:   {context_length}")

    try:
        metrics = module.validate(
            input_data,
            local_model_path=model_path,
            local_base_model_path=base_model_path,
            local_validation_file=validation_file,
        )
        logger.info(f"Validation complete! Score: {metrics.score:.4f}")
    except KeyboardInterrupt:
        click.echo("\nValidation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    finally:
        module.cleanup()


if __name__ == "__main__":
    main()
