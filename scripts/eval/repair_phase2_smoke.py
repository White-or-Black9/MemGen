import argparse
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

from common.config import Config
from data import get_data_builder
from main import set_seed
from memgen.model import MemGenModel
from memgen.runner import MemGenRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run verified original MemGen static evaluation."
    )
    parser.add_argument("--cfg-path", default="configs/latent_memory/gsm8k.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--max-response-length", type=int, default=128)
    return parser.parse_args()


def verify_adapter(model, component, checkpoint_path, adapter_name):
    runtime = get_peft_model_state_dict(component.model, adapter_name=adapter_name)
    saved = load_file(str(checkpoint_path))
    shared_keys = set(runtime) & set(saved)
    return {
        "runtime_key_count": len(runtime),
        "checkpoint_key_count": len(saved),
        "missing_keys": sorted(set(saved) - set(runtime)),
        "unexpected_keys": sorted(set(runtime) - set(saved)),
        "shape_mismatches": sorted(
            key for key in shared_keys if runtime[key].shape != saved[key].shape
        ),
        "value_mismatches": sorted(
            key
            for key in shared_keys
            if runtime[key].shape == saved[key].shape
            and not torch.equal(runtime[key].cpu(), saved[key].cpu())
        ),
    }


def install_generation_trace(model, trace):
    original_should_augment = model._should_augment
    original_generate = model.generate
    original_augment_prompt = model.weaver.augment_prompt
    original_augment_inference = model.weaver.augment_inference

    def tracked_should_augment(self, *args, **kwargs):
        trace["trigger_decision_calls"] += 1
        return original_should_augment(*args, **kwargs)

    def tracked_augment_prompt(self, *args, **kwargs):
        trace["weaver_prompt_calls"] += 1
        return original_augment_prompt(*args, **kwargs)

    def tracked_augment_inference(self, *args, **kwargs):
        trace["weaver_inference_calls"] += 1
        return original_augment_inference(*args, **kwargs)

    def tracked_generate(self, *args, **kwargs):
        kwargs["return_augmentation_mask"] = True
        torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids, augmentation_mask = original_generate(*args, **kwargs)
        torch.cuda.synchronize()
        input_ids = args[0] if args else kwargs["input_ids"]
        response_ids = output_ids[:, input_ids.size(1):].detach().cpu().contiguous()
        augmentation_mask = augmentation_mask.detach().cpu().contiguous()
        trace["generation_records"].append({
            "response_token_count": int(response_ids.ne(model.tokenizer.pad_token_id).sum()),
            "response_token_sha256": hashlib.sha256(
                response_ids.numpy().tobytes()
            ).hexdigest(),
            "augmentation_mask_sha256": hashlib.sha256(
                augmentation_mask.numpy().tobytes()
            ).hexdigest(),
            "augmentation_mask": augmentation_mask.tolist(),
            "generation_latency_seconds": time.perf_counter() - start,
        })
        return output_ids

    model._should_augment = MethodType(tracked_should_augment, model)
    model.generate = MethodType(tracked_generate, model)
    model.weaver.augment_prompt = MethodType(tracked_augment_prompt, model.weaver)
    model.weaver.augment_inference = MethodType(
        tracked_augment_inference, model.weaver
    )


def main():
    args = parse_args()
    model_path = str(Path(args.model_path).resolve())
    checkpoint_path = Path(args.checkpoint_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sample_start < 0 or args.sample_count <= 0:
        raise ValueError("sample-start must be non-negative and sample-count positive")

    config_args = SimpleNamespace(
        cfg_path=args.cfg_path,
        options=[
            "model.model_name",
            model_path,
            "model.load_model_path",
            str(checkpoint_path),
            "model.max_prompt_aug_num",
            "1",
            "model.max_inference_aug_num",
            "3",
            "model.weaver.model_name",
            model_path,
            "model.weaver.prompt_latents_len",
            "8",
            "model.weaver.inference_latents_len",
            "8",
            "model.trigger.model_name",
            model_path,
            "model.trigger.active",
            "False",
            "run.mode",
            "evaluate",
            "run.seed",
            "42",
            "run.interaction.batch_size",
            "1",
            "run.interaction.temperature",
            "0.0",
            "run.interaction.max_response_length",
            str(args.max_response_length),
            "run.interaction.weaver_do_sample",
            "False",
            "run.interaction.trigger_do_sample",
            "False",
        ],
    )
    config = Config(config_args)
    config_dict = config.to_dict()
    set_seed(42, use_gpu=True)

    data_builder = get_data_builder(config_dict["dataset"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = MemGenModel.from_config(config_dict["model"])

    adapter_results = {
        "weaver": verify_adapter(
            model,
            model.weaver,
            checkpoint_path / "weaver" / "weaver" / "adapter_model.safetensors",
            model.weaver.adapter_name,
        ),
        "trigger": verify_adapter(
            model,
            model.trigger,
            checkpoint_path / "trigger" / "trigger" / "adapter_model.safetensors",
            model.trigger.adapter_name,
        ),
    }
    adapter_results["load_warnings"] = [
        str(item.message)
        for item in caught
        if "adapter" in str(item.message).lower()
        or "missing" in str(item.message).lower()
        or "unexpected" in str(item.message).lower()
    ]
    for component_name in ("weaver", "trigger"):
        result = adapter_results[component_name]
        if any(
            result[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "shape_mismatches",
                "value_mismatches",
            )
        ):
            raise RuntimeError(f"{component_name} adapter verification failed: {result}")

    runner = MemGenRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=str(output_dir),
    )
    sample_ids = list(
        range(args.sample_start, args.sample_start + args.sample_count)
    )
    runner.test_dataset = runner.test_dataset.select(sample_ids)

    trace = {
        "trigger_active": model.trigger.active,
        "trigger_decision_calls": 0,
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
        "generation_records": [],
    }
    install_generation_trace(model, trace)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    runner.evaluate()
    torch.cuda.synchronize()
    latency = time.perf_counter() - start

    answer_path = output_dir / "evaluate" / "answer.json"
    answer_lines = answer_path.read_text(encoding="utf-8").splitlines()
    answer_records = [json.loads(line) for line in answer_lines]
    prediction_count = sum("completion" in record for record in answer_records)
    summary_count = sum("summary_metrics" in record for record in answer_records)
    prediction_records = [
        record for record in answer_records if "completion" in record
    ]
    summary_records = [
        record for record in answer_records if "summary_metrics" in record
    ]
    if prediction_count != args.sample_count or summary_count != 1:
        raise RuntimeError(
            "Unexpected answer.json structure: "
            f"predictions={prediction_count}, summaries={summary_count}"
        )
    if any(not record["completion"].strip() for record in prediction_records):
        raise RuntimeError("At least one prediction is empty")
    summary_metrics = summary_records[0]["summary_metrics"]
    if any(not math.isfinite(float(value)) for value in summary_metrics.values()):
        raise RuntimeError(f"Non-finite summary metric found: {summary_metrics}")
    if len(trace["generation_records"]) != args.sample_count:
        raise RuntimeError(
            "Generation trace count does not match sample count: "
            f"{len(trace['generation_records'])} != {args.sample_count}"
        )
    verification = {
        "config_file": str(Path(args.cfg_path).resolve()),
        "model_path": model_path,
        "checkpoint_path": str(checkpoint_path),
        "dataset": "gsm8k/main test",
        "sample_ids": sample_ids,
        "sample_count": args.sample_count,
        "seed": 42,
        "batch_size": 1,
        "decoding": {
            "temperature": 0.0,
            "max_response_length": args.max_response_length,
            "weaver_do_sample": False,
            "trigger_do_sample": False,
        },
        "adapter_verification": adapter_results,
        "generation_trace": trace,
        "answer_file": str(answer_path),
        "answer_line_count": len(answer_lines),
        "prediction_count": prediction_count,
        "summary_count": summary_count,
        "summary_metrics": summary_metrics,
        "answer_nonempty": answer_path.stat().st_size > 0,
        "latency_seconds": latency,
        "latency_per_sample_seconds": latency / args.sample_count,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    contract_dir = output_dir / "json"
    contract_dir.mkdir(parents=True, exist_ok=True)
    metric_contract = {
        "baseline_id": "memgen-gsm8k-sft-official-v1",
        "task": "GSM8K mathematical reasoning",
        "dataset": {
            "name": "gsm8k",
            "config": "main",
            "split": "test",
            "sample_ids": sample_ids,
        },
        "evaluation_path": (
            "Config -> MemGenModel.from_config -> MemGenRunner.evaluate()"
        ),
        "metrics": {
            "compute_reward": {
                "description": "Exact-answer reward computed by the official GSM8K environment",
                "direction": "higher_is_better",
                "origin_path": str(answer_path),
                "required": True,
            }
        },
        "seed": 42,
        "batch_size": 1,
        "decoding": verification["decoding"],
        "model_path": model_path,
        "checkpoint_path": str(checkpoint_path),
        "config_file": verification["config_file"],
        "known_deviations": [
            f"Uses a fixed {args.sample_count}-sample test subset rather than "
            "the full GSM8K test split."
        ],
    }
    (contract_dir / "metric_contract.json").write_text(
        json.dumps(metric_contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
