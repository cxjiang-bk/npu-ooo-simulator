import json
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
else:  # pragma: no cover - exercised by dependency-minimal environments
    torch = None

from npu_ooo.cli import _compile_from_args, build_parser
from npu_ooo.compiler.pipeline import _stablehlo_export_call
from npu_ooo.frontend import Workload, build_declarative_workload, load_workload_config
from npu_ooo.frontend import official_stablehlo_available, torch_xla_available


FRONTEND_AVAILABLE = bool(
    TORCH_AVAILABLE
    and torch_xla_available()
    and official_stablehlo_available()
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class WorkloadConfigTest(unittest.TestCase):
    EXPECTED_CONFIGS = {
        "attention-causal-factory.json",
        "attention.json",
        "bert-multilayer.json",
        "bert-oneblock.json",
        "deepseek-decode-dense-fixed-window.json",
        "deepseek-decode-moe-fixed-window.json",
        "deepseek-prefill-2layer-model-proxy.json",
        "deepseek-prefill-dense-oneblock.json",
        "deepseek-prefill-moe-oneblock.json",
        "flash-attention.json",
        "gptj-prefill-2layer-model-proxy.json",
        "gptj-prefill-oneblock.json",
        "llama-decode-fixed-window.json",
        "llama-prefill-2layer-model-proxy.json",
        "llama-prefill-oneblock.json",
        "matmul.json",
        "moe-top2.json",
        "resnet50-2block-model-proxy.json",
        "resnet50-bottleneck.json",
    }

    def test_all_shipped_configs_construct_and_run_forward(self) -> None:
        paths = tuple(sorted(Path("configs/workloads").glob("*.json")))
        self.assertTrue(self.EXPECTED_CONFIGS.issubset({path.name for path in paths}))
        for path in paths:
            with self.subTest(config=path.name):
                workload = load_workload_config(path).workload
                with torch.no_grad():
                    output = workload.module(*workload.args, **workload.kwargs)
                outputs = output if isinstance(output, tuple) else (output,)
                self.assertTrue(outputs)
                self.assertTrue(all(isinstance(item, torch.Tensor) for item in outputs))

    def test_flash_attention_uses_exact_online_softmax(self) -> None:
        workload = load_workload_config("configs/workloads/flash-attention.json").workload
        q, k, v, mask = workload.args
        with torch.no_grad():
            actual = workload.module(q, k, v, mask)
            reference = torch.matmul(
                torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
                    + mask,
                    dim=-1,
                ),
                v,
            )
        self.assertTrue(torch.allclose(actual, reference, atol=1e-6, rtol=1e-5))
        self.assertEqual(
            workload.provenance["algorithm"], "flash_attention_online_softmax"
        )
        self.assertFalse(workload.provenance["materializes_full_attention_matrix"])

    def test_top2_moe_owns_routing_and_normalizes_selected_experts(self) -> None:
        workload = load_workload_config("configs/workloads/moe-top2.json").workload
        value = workload.args[0]
        module = workload.module
        with torch.no_grad():
            weights = module.routing_weights(value)
            output = module(value)
            ranking = module.router(value) + module.ranking_tie_break
            expected_indices = torch.topk(ranking, 2, dim=-1).indices
            expected_mask = torch.zeros_like(weights, dtype=torch.bool).scatter(
                -1, expected_indices, True
            )
        self.assertEqual(torch.count_nonzero(weights).item(), value.shape[1] * 2)
        self.assertTrue(torch.equal(weights > 0, expected_mask))
        self.assertTrue(
            torch.allclose(
                torch.sum(weights, dim=-1),
                torch.ones_like(weights[..., 0]),
            )
        )
        self.assertEqual(tuple(output.shape), tuple(value.shape))
        self.assertEqual(workload.provenance["routing_contract"], "internal_router+internal_top2")
        self.assertFalse(workload.provenance["dynamic_token_compaction"])

    def test_kwargs_positionalization_does_not_append_unused_defaults(self) -> None:
        class OptionalInput(torch.nn.Module):
            def forward(self, value, scale=1.0, optional=None):
                return value * scale if optional is None else value + optional

        value = torch.ones(1)
        args, kwargs = _stablehlo_export_call(OptionalInput(), (), {"value": value})
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], value)
        self.assertIsNone(kwargs)
        args, kwargs = _stablehlo_export_call(
            OptionalInput(),
            (value,),
            {"optional": value},
        )
        self.assertEqual(len(args), 3)
        self.assertIs(args[0], value)
        self.assertEqual(args[1], 1.0)
        self.assertIs(args[2], value)
        self.assertIsNone(kwargs)

    def test_multilayer_paper_proxy_materializes_requested_depth(self) -> None:
        for filename in (
            "gptj-prefill-2layer-model-proxy.json",
            "llama-prefill-2layer-model-proxy.json",
            "deepseek-prefill-2layer-model-proxy.json",
        ):
            workload = load_workload_config(Path("configs/workloads") / filename).workload
            self.assertEqual(len(workload.module.backbone.layers), 2)

    def test_paper_config_retains_dtype_fallback_provenance(self) -> None:
        workload = load_workload_config(
            "configs/workloads/deepseek-prefill-dense-oneblock.json"
        ).workload
        paper = workload.provenance["paper_benchmark"]
        self.assertEqual(paper["spec"]["dtype"], "bfloat16")
        self.assertEqual(paper["attributes"]["simulation_dtype"], "float32")
        self.assertTrue(paper["attributes"]["dtype_fallback"])

    def test_mixed_kwargs_scalars_none_and_nested_tree_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            _write(
                path,
                {
                    "schema_version": 1,
                    "seed": 5,
                    "experiment_id": "input-contract",
                    "model": {
                        "factory": "examples.configurable_models:InputContractProbe",
                        "kwargs": {"width": 4},
                        "dtype": "float32",
                    },
                    "inputs": {
                        "kwargs": {
                            "token_ids": {
                                "kind": "tensor",
                                "shape": [1, 4],
                                "dtype": "int64",
                                "init": {"kind": "randint", "low": 0, "high": 8},
                            },
                            "mask": {
                                "kind": "tensor",
                                "shape": [1, 4],
                                "dtype": "bool",
                                "init": {"kind": "randint", "low": 0, "high": 2},
                            },
                            "scale": 0.5,
                            "optional": None,
                            "nested": {
                                "kind": "tuple",
                                "items": [
                                    {
                                        "kind": "list",
                                        "items": [
                                            {
                                                "kind": "tensor",
                                                "shape": [1, 4],
                                                "dtype": "float32",
                                                "init": "ones",
                                            }
                                        ],
                                    },
                                    {
                                        "kind": "dict",
                                        "items": {
                                            "bias": {
                                                "kind": "tensor",
                                                "shape": [1, 4],
                                                "dtype": "float32",
                                                "init": "zeros",
                                            }
                                        },
                                    },
                                ],
                            },
                        }
                    },
                },
            )

            loaded = load_workload_config(path)
            workload = loaded.workload
            self.assertEqual(workload.experiment_id, "input-contract")
            self.assertEqual(workload.kwargs["token_ids"].dtype, torch.int64)
            self.assertEqual(workload.kwargs["mask"].dtype, torch.bool)
            self.assertIsNone(workload.kwargs["optional"])
            self.assertIsInstance(workload.kwargs["nested"], tuple)
            self.assertIsInstance(workload.kwargs["nested"][0], list)
            self.assertIsInstance(workload.kwargs["nested"][1], dict)
            self.assertEqual(tuple(workload.module(**workload.kwargs).shape), (1, 4))
            self.assertEqual(
                workload.input_signature["kwargs"]["token_ids"]["forward_argument"],
                "token_ids",
            )

    def test_seed_controls_module_and_input_generation(self) -> None:
        first = load_workload_config("configs/workloads/bert-multilayer.json").workload
        second = load_workload_config("configs/workloads/bert-multilayer.json").workload
        self.assertTrue(
            torch.equal(
                next(first.module.parameters()),
                next(second.module.parameters()),
            )
        )
        self.assertTrue(torch.equal(first.kwargs["input_ids"], second.kwargs["input_ids"]))
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(
                Path("configs/workloads/bert-multilayer.json").read_text(encoding="utf-8")
            )
            payload["seed"] += 1
            changed = load_workload_config(
                _write(Path(directory) / "changed-seed.json", payload)
            ).workload
        self.assertFalse(
            torch.equal(next(first.module.parameters()), next(changed.module.parameters()))
        )

    def test_outer_seed_controls_paper_workload_factory(self) -> None:
        path = Path("configs/workloads/gptj-prefill-oneblock.json")
        first = load_workload_config(path).workload
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seed"] += 1
        with tempfile.TemporaryDirectory() as directory:
            changed = load_workload_config(
                _write(Path(directory) / path.name, payload)
            ).workload
        self.assertFalse(
            torch.equal(
                dict(first.module.named_parameters())["q_proj.weight"],
                dict(changed.module.named_parameters())["q_proj.weight"],
            )
        )

    def test_constructor_parameters_materialize_real_layers(self) -> None:
        workload = load_workload_config("configs/workloads/bert-multilayer.json").workload
        self.assertEqual(workload.module.num_layers, 2)
        self.assertEqual(len(workload.module.layers), 2)
        small_parameter_count = sum(item.numel() for item in workload.module.parameters())
        other = type(workload.module)(
            num_layers=3,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            vocab_size=32,
            max_position_embeddings=8,
        )
        self.assertEqual(len(other.layers), 3)
        self.assertGreater(sum(item.numel() for item in other.parameters()), small_parameter_count)

    def test_python_factory_returns_generic_workload(self) -> None:
        loaded = load_workload_config("configs/workloads/attention-causal-factory.json")
        self.assertIsInstance(loaded.workload, Workload)
        self.assertEqual(loaded.workload.provenance["construction"], "python_workload_factory")
        self.assertEqual(set(loaded.workload.kwargs), {"attention_mask"})

    def test_paper_builder_layers_metadata_over_generic_workload(self) -> None:
        from examples.paper_benchmarks import build_paper_benchmark

        paper = build_paper_benchmark("bert-base", variant="micro")
        self.assertIsInstance(paper.workload, Workload)
        self.assertEqual(paper.spec.case_id, "bert-base")
        self.assertNotIn("case_id", paper.workload.provenance)

    def test_scalar_tensor_is_distinct_from_python_scalar(self) -> None:
        workload = build_declarative_workload(
            {
                "factory": "examples.configurable_models:ScalarTensorProbe",
                "dtype": "float32",
            },
            {
                "args": [
                    {
                        "kind": "tensor",
                        "shape": [],
                        "dtype": "float32",
                        "init": {"kind": "explicit", "values": 3.0},
                    },
                    2.0,
                ]
            },
        )
        self.assertEqual(workload.args[0].shape, torch.Size([]))
        self.assertIsInstance(workload.args[1], float)
        self.assertEqual(workload.module(*workload.args).item(), 6.0)
        self.assertEqual(workload.input_signature["args"][0]["kind"], "tensor")
        self.assertEqual(workload.input_signature["args"][1]["kind"], "python_scalar")

    def test_relative_tensor_data_path_is_resolved_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "lhs.json", {"values": [[1, 2], [3, 4]]})
            path = _write(
                root / "workload.json",
                {
                    "schema_version": 1,
                    "model": {
                        "factory": "examples.configurable_models:Matmul",
                        "dtype": "float32",
                    },
                    "inputs": {
                        "args": [
                            {
                                "kind": "tensor",
                                "shape": [2, 2],
                                "dtype": "float32",
                                "init": {"kind": "file", "path": "lhs.json"},
                            },
                            {
                                "kind": "tensor",
                                "shape": [2, 2],
                                "dtype": "float32",
                                "init": "ones",
                            },
                        ]
                    },
                    "compile": {"machine_config": "machine.json"},
                },
            )
            loaded = load_workload_config(path)
            self.assertEqual(loaded.workload.args[0][1, 1].item(), 4)
            self.assertEqual(
                loaded.compile_options["machine_config"],
                str((root / "machine.json").resolve()),
            )
            init = loaded.workload.input_signature["args"][0]["init"]
            self.assertEqual(init["resolved_path"], str((root / "lhs.json").resolve()))

    def test_validation_reports_exact_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "schema_version": 1,
                "model": {
                    "factory": "examples.configurable_models:Matmul",
                    "dtype": "float32",
                },
                "inputs": {
                    "args": [
                        {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "complex64",
                            "init": "zeros",
                        },
                        {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "float32",
                            "init": "ones",
                        },
                    ]
                },
            }
            path = _write(root / "invalid.json", base)
            with self.assertRaisesRegex(ValueError, r"inputs\.args\[0\]\.dtype"):
                load_workload_config(path)
            base["inputs"]["args"][0]["dtype"] = "float32"
            base["inputs"]["args"][0]["init"] = {
                "kind": "explicit",
                "values": [1, 2, 3],
            }
            _write(path, base)
            with self.assertRaisesRegex(ValueError, "shape requires 4"):
                load_workload_config(path)

    def test_unknown_and_conflicting_workload_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            _write(path, {"schema_version": 1, "unknown": 1})
            with self.assertRaisesRegex(ValueError, r"config: unknown field\(s\): unknown"):
                load_workload_config(path)
            _write(
                path,
                {
                    "schema_version": 1,
                    "model": {"factory": "examples.configurable_models:Matmul"},
                    "inputs": {"args": []},
                    "workload": {
                        "factory": "examples.configurable_models:build_attention_workload"
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "declare exactly one"):
                load_workload_config(path)

    def test_constructor_forward_and_random_range_errors_are_attributed(self) -> None:
        model = {
            "factory": "examples.configurable_models:Matmul",
            "dtype": "float32",
        }
        one_input = {
            "args": [
                {
                    "kind": "tensor",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "init": "ones",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, r"inputs: do not match forward"):
            build_declarative_workload(model, one_input)
        with self.assertRaisesRegex(ValueError, r"model\.kwargs"):
            build_declarative_workload({**model, "kwargs": {"missing": 1}}, one_input)
        integer_input = {
            "args": [
                {
                    "kind": "tensor",
                    "shape": [2, 2],
                    "dtype": "int64",
                    "init": {"kind": "randint", "low": 3, "high": 3},
                },
                {
                    "kind": "tensor",
                    "shape": [2, 2],
                    "dtype": "int64",
                    "init": "ones",
                },
            ]
        }
        with self.assertRaisesRegex(ValueError, r"inputs\.args\[0\]\.init"):
            build_declarative_workload(model, integer_input)


@unittest.skipUnless(TORCH_AVAILABLE, "requires PyTorch")
class WorkloadCliResolutionTest(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        return _write(
            root / "workload.json",
            {
                "schema_version": 1,
                "model": {
                    "factory": "examples.configurable_models:Matmul",
                    "dtype": "float32",
                },
                "inputs": {
                    "args": [
                        {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "float32",
                            "init": "ones",
                        },
                        {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "float32",
                            "init": "ones",
                        },
                    ]
                },
                "compile": {"arch": "lpu-like", "tile_size": 2},
            },
        )

    def test_json_beats_argparse_defaults_and_explicit_cli_beats_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config(Path(directory))
            with patch("npu_ooo.cli.compile_torch_module", return_value=object()):
                args = build_parser().parse_args(["compile", "--config", str(path)])
                _compiled, machine, _workload, resolved, _factory = _compile_from_args(args)
                self.assertEqual(machine.config_id, "lpu-like")
                self.assertEqual(resolved["compile"]["tile_size"], 2)

                args = build_parser().parse_args(
                    ["compile", "--config", str(path), "--arch", "minimal", "--tile-size", "4"]
                )
                _compiled, machine, _workload, resolved, _factory = _compile_from_args(args)
                self.assertEqual(machine.config_id, "minimal")
                self.assertEqual(resolved["compile"]["tile_size"], 4)

    def test_explicit_arch_replaces_machine_file_inherited_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config(Path(directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["compile"]["machine_config"] = "does-not-exist.json"
            _write(path, payload)
            args = build_parser().parse_args(
                ["compile", "--config", str(path), "--arch", "minimal"]
            )
            with patch("npu_ooo.cli.compile_torch_module", return_value=object()):
                _compiled, machine, _workload, resolved, _factory = _compile_from_args(args)
            self.assertEqual(machine.config_id, "minimal")
            self.assertIsNone(resolved["compile"]["machine_config"])

    def test_config_and_legacy_inputs_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._config(Path(directory))
            args = build_parser().parse_args(
                ["compile", "--config", str(path), "--input-shape", "2,2"]
            )
            with self.assertRaisesRegex(ValueError, "one workload input source"):
                _compile_from_args(args)

    def test_compile_and_sim_runtime_section_is_resolved(self) -> None:
        args = build_parser().parse_args(
            [
                "compile-and-sim",
                "--config",
                "configs/workloads/attention-causal-factory.json",
                "--policy",
                "static_pipeline",
            ]
        )
        with patch("npu_ooo.cli.compile_torch_module", return_value=object()):
            _compiled, _machine, _workload, resolved, _factory = _compile_from_args(args)
        self.assertEqual(args.event_backend, "cycle_event")
        self.assertTrue(args.address_scoreboard)
        self.assertEqual(args.policy, "static_pipeline")
        self.assertEqual(resolved["runtime"]["policy"], "static_pipeline")


@unittest.skipUnless(FRONTEND_AVAILABLE, "requires PyTorch, Torch-XLA and official StableHLO")
class WorkloadFrontendIntegrationTest(unittest.TestCase):
    def test_flash_attention_and_top2_moe_reach_scheduler_visible_tisa(self) -> None:
        from npu_ooo.cli import main

        cases = (
            ("flash-attention.json", "flash_attention"),
            ("moe-top2.json", "moe"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for filename, family in cases:
                with self.subTest(workload=filename):
                    output = Path(directory) / family
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "compile",
                                    "--config",
                                    str(Path("configs/workloads") / filename),
                                    "--output-dir",
                                    str(output),
                                ]
                            ),
                            0,
                        )
                    graph = json.loads(
                        (output / "01_gc" / "canonical_graph.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    regions = [
                        item
                        for item in graph["attributes"].get("semantic_regions", [])
                        if item.get("semantic_family") == family
                    ]
                    self.assertEqual(len(regions), 1)
                    self.assertFalse(regions[0]["opaque"])
                    self.assertEqual(
                        regions[0]["scheduler_visibility"],
                        "member_tisa_instructions",
                    )
                    tisa = json.loads(
                        (output / "03_tisa" / "tisa_program.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertGreater(len(tisa["instructions"]), 1)
                    self.assertTrue(
                        any(
                            item["attributes"].get("semantic_region_family")
                            == family
                            for item in tisa["instructions"]
                        )
                    )
                    if family == "flash_attention":
                        self.assertEqual(regions[0]["query_block_count"], 2)
                        self.assertEqual(regions[0]["kv_block_count"], 1)
                        self.assertEqual(regions[0]["score_block_count"], 2)
                        self.assertFalse(
                            regions[0]["materializes_full_attention_matrix"]
                        )
                    else:
                        self.assertEqual(regions[0]["top_k"], 2)
                        self.assertEqual(regions[0]["expert_count"], 4)
                        targets = {
                            item["attributes"].get("frontend_target")
                            for item in graph["operators"]
                        }
                        self.assertIn("stablehlo.compare", targets)
                        self.assertIn("stablehlo.select", targets)

    def test_flash_attention_long_sequences_compile_with_sub_head_tile(self) -> None:
        from examples.configurable_models import build_flash_attention_workload
        from npu_ooo.arch import lpu_like_machine_config
        from npu_ooo.compiler import compile_torch_module

        workload = build_flash_attention_workload(
            batch_size=1,
            num_heads=2,
            query_length=16,
            key_length=24,
            head_dim=16,
            query_block_size=4,
            kv_block_size=6,
            causal=True,
            dtype="float32",
        )
        compiled = compile_torch_module(
            workload.module,
            workload.args,
            lpu_like_machine_config(),
            model_id="flash-attention-long-sequence",
            tile_size=4,
        )
        self.assertEqual(compiled.validate(), ())
        transform_tasks = [
            task
            for task in compiled.backend_artifact.execution_graph.tasks
            if task.attributes.get("semantic_family") in {"slice", "reshape", "concatenate"}
        ]
        self.assertTrue(transform_tasks)
        self.assertTrue(
            all(task.attributes.get("transform_granularity") == "output_tile" for task in transform_tasks)
        )
        self.assertGreater(len(compiled.tile_graph.tiles), 16)

    def test_gptj_model_proxy_recovers_layernorm_after_embedding_reshape(self) -> None:
        from npu_ooo.cli import main

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            self.assertEqual(
                main(
                    [
                        "compile",
                        "--config",
                        "configs/workloads/gptj-prefill-2layer-model-proxy.json",
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            graph = json.loads(
                (output / "01_gc" / "canonical_graph.json").read_text(encoding="utf-8")
            )
            operation_types = [item["op_type"] for item in graph["operators"]]
            self.assertEqual(operation_types.count("layernorm"), 4)
            self.assertNotIn("stablehlo.batch_norm_training", operation_types)

    def test_multilayer_integer_input_and_fixed_decode_compile_officially(self) -> None:
        from npu_ooo.cli import main

        cases = (
            (
                "configs/workloads/attention.json",
                "json-attention",
                "q",
                "float32",
            ),
            (
                "configs/workloads/bert-multilayer.json",
                "bert-proxy-2layer-static",
                "input_ids",
                "int64",
            ),
            (
                "configs/workloads/llama-decode-fixed-window.json",
                "llama-decode-seqlen1-window4",
                "x",
                "float16",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (config, model_id, input_name, input_dtype) in enumerate(cases):
                output = Path(directory) / str(index)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        main(["compile", "--config", config, "--output-dir", str(output)]),
                        0,
                    )
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["model_id"], model_id)
                stablehlo = json.loads(
                    (output / "00_frontend" / "stablehlo_module.json").read_text(encoding="utf-8")
                )
                self.assertTrue(stablehlo["verified"])
                signature = json.loads(
                    (output / "00_frontend" / "input_signature.json").read_text(encoding="utf-8")
                )
                self.assertEqual(signature["kwargs"][input_name]["dtype"], input_dtype)
                self.assertTrue((output / "04_backend" / "backend_artifact.json").is_file())

    def test_python_workload_factory_compile_and_sim_uses_json_runtime(self) -> None:
        from npu_ooo.cli import main

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            self.assertEqual(
                main(
                    [
                        "compile-and-sim",
                        "--config",
                        "configs/workloads/attention.json",
                        "--output-dir",
                        str(output),
                    ]
                ),
                0,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["event_backend"], "cycle_event")
            self.assertEqual(manifest["policy"], "dynamic_ready_queue")
            self.assertGreater(manifest["total_cycles"], 0)
            resolved = json.loads(
                (output / "00_frontend" / "resolved_workload_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(resolved["runtime"]["event_backend"], "cycle_event")


if __name__ == "__main__":
    unittest.main()
