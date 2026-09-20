r"""MNIST convergence: synchronous 1F1B, delayed 2BW, and update-level EF.

    python examples/mnist_pipedream_2bw.py --devices cuda:0,cuda:1,cuda:2,cuda:3

Defaults: Adam, Lion, Muon and Dion3, five epochs, identical initial weights and
batch order within each seed. Use --seeds 0,1,2 for repeated runs, --bf16 for autocast, or
--devices cpu,cpu --epochs 1 --train-limit 512 for a small CPU smoke run.

Only a SEPARATE fresh-model replay is profiled: --profile-warmup updates then
--profile-steps updates (default 3). No profiling or per-update weight snapshots
run during convergence training. The replay does not change convergence weights,
optimizer counters, data order, or timings. Traces are gzip-compressed.

Adam here means Adam with coupled L2 decay, NOT AdamW; select --optimizers adamw
for decoupled decay. Both plain and EF variants use AdamEF with c=0/c=1, so
correction is the only implementation difference. --adam-triton opts Adam/AdamW
into fused CUDA updates; --lion-triton does the same for Lion. Both default to
eager updates, require CUDA when selected, and work with or without EF.
Lion uses decoupled decay. Muon and Dion3 use hidden block matrices only, with
AdamW fallback for the stem, head and biases. Orthogonalization is independent
of model autocast: Muon defaults to FP32; Dion3 (upstream NorDion2 alias) defaults
to BF16 Polar Express, with optional --dion3-triton on CUDA. Dion3's compression
error feedback is always active and is distinct from optional update-level EF.
Default decay is zero for all. This is a toy comparison, not a reproduction of
the LLM recipe or evidence EF must win.
"""
from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import math
import sys
import time
import warnings
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

# Reference installs deterministic CUDA environment variables before torch.
if __package__:
    from .pipedream_2bw_reference import configure_determinism, ResidualMLPBlock
    from .pipedream_2bw_demo import baseline_stage_spans
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipedream_2bw_reference import configure_determinism, ResidualMLPBlock
    from pipedream_2bw_demo import baseline_stage_spans

import torch
from torch import nn
from torch.nn import functional as F

from ramtorch import Pipeline, AdamEF, Lion, Muon, Dion3
from ramtorch.pipeline_2bw_trace import TraceCapture, inspect_trace

MODES = ("sync", "2bw", "2bw_ef")


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--optimizers", nargs="+", choices=("adam", "adamw", "lion", "muon", "dion3"),
                        default=["adam", "lion", "muon", "dion3"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", default="0", help="comma-separated model/data seeds")
    parser.add_argument("--epochs", type=positive, default=5)
    parser.add_argument("--dim", type=positive, default=256)
    parser.add_argument("--blocks", type=positive, default=4)
    parser.add_argument("--batch-size", type=positive, default=256, help="full optimizer-update batch")
    parser.add_argument("--microbatches", type=positive, default=4)
    parser.add_argument("--eval-batch-size", type=positive, default=512)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--adamw-lr", type=float, default=1e-3)
    parser.add_argument("--lion-lr", type=float, default=1e-4)
    parser.add_argument("--adam-weight-decay", type=float, default=0.)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.)
    parser.add_argument("--lion-weight-decay", type=float, default=0.)
    parser.add_argument("--adam-triton", action="store_true",
                        help="use fused CUDA Triton updates for selected adam/adamw variants, with or without EF")
    parser.add_argument("--lion-triton", action="store_true",
                        help="use fused CUDA Triton updates for selected lion variants, with or without EF")
    parser.add_argument("--muon-lr", type=float, default=.02)
    parser.add_argument("--muon-adamw-lr", type=float, default=1e-3,
                        help="Muon fallback LR for stem/head/biases")
    parser.add_argument("--muon-weight-decay", type=float, default=0.)
    parser.add_argument("--muon-momentum", type=float, default=.95)
    parser.add_argument("--muon-ns-steps", type=positive, default=5)
    parser.add_argument("--muon-ns-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--dion3-lr", type=float, default=.01)
    parser.add_argument("--dion3-adamw-lr", type=float, default=1e-3,
                        help="Dion3 fallback LR for stem/head/biases")
    parser.add_argument("--dion3-fraction", type=float, default=.25)
    parser.add_argument("--dion3-mu", type=float, default=.95)
    parser.add_argument("--dion3-beta2", type=float, default=.95,
                        help="Dion3 selected-row variance EMA coefficient")
    parser.add_argument("--dion3-weight-decay", type=float, default=0.)
    parser.add_argument("--dion3-ns-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--dion3-triton", action="store_true",
                        help="use vendored Dion3 Triton kernels (CUDA, BF16 orthogonalization only)")
    parser.add_argument("--ef-coefficient", type=float, default=1.)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu-threads", type=positive, default=4)
    parser.add_argument("--validation-size", type=positive, default=5000)
    parser.add_argument("--train-limit", type=positive, help="optional subset, smoke tests only")
    parser.add_argument("--profile-steps", type=int, default=3, help="separate replay capture; 0 disables")
    parser.add_argument("--profile-warmup", type=positive, default=3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("scratchpad/pipedream_2bw/runs/mnist"))
    args = parser.parse_args(argv)
    try:
        devices = [torch.device(d.strip()) for d in args.devices.split(",")]
        seeds = [int(seed) for seed in args.seeds.split(",")]
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error("seeds must be distinct integers in [0, 2**32)")
    if {d.type for d in devices} not in ({"cpu"}, {"cuda"}):
        parser.error("use all CPU or all CUDA stages")
    if devices[0].type == "cuda":
        if any(d.index is None for d in devices) or len(set(devices)) != len(devices):
            parser.error("use distinct indexed CUDA devices")
        if not torch.cuda.is_available() or any(d.index >= torch.cuda.device_count() for d in devices):
            parser.error("requested CUDA devices are unavailable")
        if args.bf16 or ("dion3" in args.optimizers and args.dion3_ns_dtype == "bfloat16"):
            for device in devices:
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported():
                        parser.error("BF16 unsupported on {}".format(device))
    if args.blocks < len(devices):
        parser.error("blocks must be at least the number of stages")
    if args.microbatches < len(devices) or args.batch_size % args.microbatches:
        parser.error("microbatches must be >= stages and divide batch-size")
    if args.profile_steps < 0 or args.validation_size >= 60000:
        parser.error("profile-steps must be >=0 and validation-size <60000")
    for key in ("muon_momentum", "dion3_mu", "dion3_beta2"):
        if not math.isfinite(getattr(args, key)) or not 0 <= getattr(args, key) < 1:
            parser.error(key.replace("_", "-") + " must be finite and in [0,1)")
    if not math.isfinite(args.dion3_fraction) or not 0 < args.dion3_fraction <= 1:
        parser.error("dion3-fraction must be finite and in (0,1]")
    if args.dion3_triton and (devices[0].type != "cuda" or args.dion3_ns_dtype != "bfloat16"):
        parser.error("dion3-triton requires CUDA devices and dion3-ns-dtype=bfloat16")
    if devices[0].type != "cuda":
        for flag in ("adam_triton", "lion_triton"):
            if getattr(args, flag):
                parser.error(flag.replace("_", "-") + " requires CUDA devices")
    for key in ("adam_lr", "adamw_lr", "lion_lr", "adam_weight_decay", "adamw_weight_decay",
                "lion_weight_decay", "ef_coefficient", "muon_lr", "muon_adamw_lr",
                "muon_weight_decay", "dion3_lr", "dion3_adamw_lr", "dion3_weight_decay"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(key + " must be finite and nonnegative")
    if len(set(args.modes)) != len(args.modes) or len(set(args.optimizers)) != len(args.optimizers):
        parser.error("duplicate modes/optimizers are not allowed")
    return args, devices, seeds


def make_model(args, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        # Each shard has at least one trainable residual block. No dropout,
        # batchnorm, mutable buffers, or tied weights in the resident 2BW path.
        modules = [("stem", nn.Sequential(nn.Linear(784, args.dim), nn.GELU()))]
        modules += [("block{}".format(i), ResidualMLPBlock(args.dim)) for i in range(args.blocks)]
        modules += [("head", nn.Linear(args.dim, 10))]
        return nn.Sequential(OrderedDict(modules))


def make_pipe(args, devices, seed):
    model = make_model(args, seed)
    modules = []
    for i in range(len(devices)):
        lo, hi = args.blocks * i // len(devices), args.blocks * (i+1) // len(devices)
        names = (["stem"] if i == 0 else []) + ["block{}".format(j) for j in range(lo, hi)]
        if i == len(devices)-1:
            names.append("head")
        modules.append(nn.Sequential(OrderedDict((name, model.get_submodule(name)) for name in names)))
    return Pipeline(stage_modules=modules, devices=devices, offload=False, overlap=True,
                    autocast=torch.bfloat16 if args.bf16 else None)


def factory_for(args, name, mode, stage=None):
    coefficient = args.ef_coefficient if mode == "2bw_ef" else 0.
    if name in ("adam", "adamw"):
        return functools.partial(AdamEF, lr=getattr(args, name + "_lr"),
                                 weight_decay=getattr(args, name + "_weight_decay"),
                                 betas=(.9, .999), ef_coefficient=coefficient,
                                 decoupled_weight_decay=name == "adamw", use_triton=args.adam_triton)
    if name == "lion":
        return functools.partial(Lion, lr=args.lion_lr, weight_decay=args.lion_weight_decay,
                                 betas=(.9, .99), ef_coefficient=coefficient, use_triton=args.lion_triton)
    if name not in ("muon", "dion3") or stage is None:
        raise ValueError("Muon/Dion3 factory needs its stage's named parameter partition")
    # Persist a POSITIONAL role list, not Parameter references. A CPU-optimizer
    # factory may receive new master Parameters in this same named order.
    use_matrix = [key.startswith("block") and parameter.ndim == 2
                  for key, parameter in stage.module.named_parameters()]

    def build(parameters):
        parameters = list(parameters)
        if len(parameters) != len(use_matrix):
            raise ValueError("stage factory parameter order/size changed")
        groups = []
        for role, lr in ((True, getattr(args, name + "_lr")),
                         (False, getattr(args, name + "_adamw_lr"))):
            selected = [p for p, flag in zip(parameters, use_matrix) if flag == role]
            if selected:
                group = dict(params=selected, lr=lr)
                if name == "muon":
                    group["use_muon"] = role
                else:
                    group["algorithm"] = "dion3" if role else "adamw"
                groups.append(group)
        if name == "dion3":
            return Dion3(groups, lr=args.dion3_lr, fraction=args.dion3_fraction,
                         mu=args.dion3_mu, muon_beta2=args.dion3_beta2, betas=(.9, .95),
                         weight_decay=args.dion3_weight_decay, epsilon=1e-8,
                         adjust_lr="spectral_norm", ef_coefficient=coefficient,
                         ns_dtype=getattr(torch, args.dion3_ns_dtype), use_triton=args.dion3_triton)
        return Muon(groups, momentum=args.muon_momentum, nesterov=True,
                    ns_steps=args.muon_ns_steps, ns_dtype=getattr(torch, args.muon_ns_dtype),
                    adjust_lr_fn="original", weight_decay=args.muon_weight_decay,
                    ef_coefficient=coefficient, betas=(.9, .999))
    return build


def load_data(args):
    # torchvision is an example dependency, not a RamTorch runtime dependency.
    from torchvision.datasets import MNIST
    train = MNIST(args.data_dir, train=True, download=not args.no_download)
    test = MNIST(args.data_dir, train=False, download=not args.no_download)
    split = torch.randperm(len(train), generator=torch.Generator().manual_seed(1729))
    validation_ids, train_ids = split[:args.validation_size], split[args.validation_size:]
    if args.train_limit:
        train_ids = train_ids[:args.train_limit]
    if len(train_ids) < args.batch_size:
        raise ValueError("training subset must contain at least one full batch")
    return train, test, train_ids, validation_ids


def batch_tensors(dataset, ids, pin):
    # Keep the dataset uint8 and materialize only one normalized group at a time.
    x = dataset.data[ids].reshape(-1, 784).float().div_(255.).sub_(.1307).div_(.3081)
    y = dataset.targets[ids]
    return (x.pin_memory(), y.pin_memory()) if pin else (x, y)


def epoch_batches(dataset, ids, args, seed, epoch, pin):
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed + 10000 + epoch))
    for start in range(0, len(ids)-args.batch_size+1, args.batch_size):
        yield batch_tensors(dataset, ids[order[start:start+args.batch_size]], pin)


def replay_batches(dataset, ids, args, seed, pin):
    epoch = 0
    while True:
        yield from epoch_batches(dataset, ids, args, seed, epoch, pin)
        epoch += 1


def synchronize(devices):
    for device in set(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


class MeanTrainingLoss:
    """One detached scalar accumulator; no per-microbatch host synchronization."""
    def reset(self):
        self.total = None
        self.count = 0

    def __init__(self):
        self.reset()

    def __call__(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        with torch.no_grad():
            if self.total is None:
                self.total = torch.zeros_like(loss)
            self.total.add_(loss.detach())
            self.count += 1
        return loss

    def mean(self):
        return self.total.item() / self.count


def run_sync(pipe, optimizers, loader, updates, m, loss_fn, profile_path=None):
    context = {"group": 0}
    iterator = iter(loader)
    with TraceCapture(profile_path) as trace:
        with baseline_stage_spans(pipe, trace, profile_path is not None, context):
            try:
                for group in range(updates):
                    context["group"] = group
                    with trace.span("loader_next", group=group):
                        x, y = next(iterator)
                    with trace.span("step", group=group):
                        result = pipe.step(x, targets=y, n_microbatches=m,
                                           schedule="staggered_1b1f", loss_fn=loss_fn)
                    with trace.span("flush_grads", group=group):
                        result.flush_grads()
                    for stage, optimizer in enumerate(optimizers):
                        with trace.span("U", stage=stage, group=group):
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    del result, x, y
            finally:
                synchronize(pipe.devices)


@torch.no_grad()
def evaluate(pipe, evaluation, dataset, ids, args, device):
    # Only called after run() drains; never mutate live trainer Parameters or
    # reset the 2BW session between epochs. Evaluate LATEST, not stale weights.
    live = {name: value for stage in pipe.stages for name, value in stage.module.state_dict().items()}
    evaluation.load_state_dict(live, strict=True)
    evaluation.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    for start in range(0, len(ids), args.eval_batch_size):
        x, y = batch_tensors(dataset, ids[start:start+args.eval_batch_size], False)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
            logits = evaluation(x)
            loss_sum.add_(F.cross_entropy(logits, y, reduction="sum"))
        correct.add_((logits.argmax(dim=1) == y).sum())
    loss, accuracy = loss_sum.item()/len(ids), correct.item()/len(ids)
    if not math.isfinite(loss):
        raise RuntimeError("nonfinite evaluation loss")
    return {"loss": loss, "accuracy": accuracy, "samples": len(ids)}


def state_bytes(optimizers):
    result = {"optimizer_tensor_bytes": 0, "ef_history_bytes": 0,
              "optimizer_tensor_bytes_by_state": {}, "optimizer_groups": []}
    for stage, optimizer in enumerate(optimizers):
        for group in optimizer.param_groups:
            options = {key: str(value) if isinstance(value, torch.dtype) else value
                       for key, value in group.items() if key != "params"}
            result["optimizer_groups"].append(dict(
                stage=stage, optimizer=type(optimizer).__name__, options=options,
                parameter_tensors=len(group["params"]),
                parameter_elements=sum(p.numel() for p in group["params"])))
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    size = value.numel() * value.element_size()
                    result["optimizer_tensor_bytes"] += size
                    by_state = result["optimizer_tensor_bytes_by_state"]
                    by_state[name] = by_state.get(name, 0) + size
                    if name == "previous_update":
                        result["ef_history_bytes"] += size
    return result


def train_variant(args, devices, seed, optimizer_name, mode, data):
    train, test, train_ids, validation_ids = data
    pipe = make_pipe(args, devices, seed)
    trainer = None
    loss_fn = MeanTrainingLoss()
    factories = [factory_for(args, optimizer_name, mode, stage) for stage in pipe.stages]
    # A separate evaluation copy avoids violating trainer ownership. Its memory
    # is not included in a claimed pipeline memory benchmark (none is made).
    evaluation = make_model(args, seed).to(devices[0])
    updates = len(train_ids) // args.batch_size
    pin = devices[0].type == "cuda"
    curve = []
    training_seconds = 0.
    try:
        if mode == "sync":
            optimizers = [factory(stage.module.parameters())
                          for factory, stage in zip(factories, pipe.stages)]
        else:
            trainer = pipe.train_session(optimizer_factory=factories, n_microbatches=args.microbatches,
                                         loss_fn=loss_fn)
            optimizers = trainer.optimizers
        initial = evaluate(pipe, evaluation, train, validation_ids, args, devices[0])
        curve.append(dict(epoch=0, updates=0, samples_seen=0, train_loss=None,
                          validation=initial, training_seconds=0.))
        for epoch in range(args.epochs):
            loss_fn.reset()
            loader = epoch_batches(train, train_ids, args, seed, epoch, pin)
            synchronize(devices)
            start = time.perf_counter()
            if trainer is None:
                run_sync(pipe, optimizers, loader, updates, args.microbatches, loss_fn)
            else:
                trainer.run(loader, updates=updates)
            synchronize(devices)
            elapsed = time.perf_counter()-start
            training_seconds += elapsed
            mean_loss = loss_fn.mean()
            if not math.isfinite(mean_loss):
                raise RuntimeError("nonfinite training loss")
            validation = evaluate(pipe, evaluation, train, validation_ids, args, devices[0])
            curve.append(dict(epoch=epoch+1, updates=(epoch+1)*updates,
                              samples_seen=(epoch+1)*updates*args.batch_size, train_loss=mean_loss,
                              validation=validation, training_seconds=training_seconds))
            print("{} {} seed={} epoch={}/{} train={:.4f} val={:.4f} acc={:.2%} train_s={:.2f}".format(
                optimizer_name, mode, seed, epoch+1, args.epochs, mean_loss, validation["loss"],
                validation["accuracy"], elapsed), flush=True)
        heldout = evaluate(pipe, evaluation, test, torch.arange(len(test)), args, devices[0])
        params = sum(p.numel()*p.element_size() for stage in pipe.stages for p in stage.module.parameters())
        return dict(seed=seed, optimizer=optimizer_name, mode=mode, curve=curve,
                    test=heldout, training_seconds=training_seconds, parameter_bytes=params,
                    compute_weight_banks=1 if mode == "sync" else 2,
                    muon_parameter_elements=(sum(p.numel() for optimizer in optimizers
                                                for group in optimizer.param_groups
                                                if group.get("use_muon", False)
                                                for p in group["params"])),
                    dion3_parameter_elements=(sum(p.numel() for optimizer in optimizers
                                                 for group in optimizer.param_groups
                                                 if group.get("algorithm") == "dion3"
                                                 for p in group["params"])),
                    **state_bytes(optimizers))
    finally:
        if trainer is not None:
            trainer.close()
        pipe.close()


def profile_variant(args, devices, seed, optimizer_name, mode, data):
    train, _, ids, _ = data
    pipe = make_pipe(args, devices, seed)
    factories = [factory_for(args, optimizer_name, mode, stage) for stage in pipe.stages]
    trainer = None
    path = args.output_dir / ("{}_{}_seed{}_profile.json.gz".format(optimizer_name, mode, seed))
    loader = replay_batches(train, ids, args, seed, devices[0].type == "cuda")
    try:
        if mode == "sync":
            optimizers = [factory(stage.module.parameters())
                          for factory, stage in zip(factories, pipe.stages)]
            run_sync(pipe, optimizers, loader, args.profile_warmup, args.microbatches, F.cross_entropy)
            run_sync(pipe, optimizers, loader, args.profile_steps, args.microbatches, F.cross_entropy, path)
        else:
            trainer = pipe.train_session(optimizer_factory=factories, n_microbatches=args.microbatches,
                                         loss_fn=F.cross_entropy)
            trainer.run(loader, updates=args.profile_warmup)
            trainer.run(loader, updates=args.profile_steps, profile_path=path)
        inspection = inspect_trace(path)
        expected = {"F": args.profile_steps*args.microbatches*len(devices),
                    "B": args.profile_steps*args.microbatches*len(devices),
                    "U": args.profile_steps*len(devices)}
        counts = inspection["cpu_annotations_by_name"]
        if any(counts.get(key, 0) != value for key, value in expected.items()):
            raise AssertionError("bounded profile operation count mismatch: {}".format(counts))
        if any(device.type == "cuda" and not inspection["gpu_kernels_per_device"].get(str(device.index))
               for device in devices):
            raise AssertionError("profile is missing real CUDA kernels on a stage")
        print("PROFILE {}: {} updates, F/B/U verified".format(path.name, args.profile_steps), flush=True)
        return dict(path=path.name, captured_updates=args.profile_steps,
                    preceding_warmup_updates=args.profile_warmup, inspection=inspection)
    finally:
        if trainer is not None:
            trainer.close()
        pipe.close()


def save_results(args, report):
    with (args.output_dir / "metrics.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output_dir / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["optimizer", "mode", "seed", "epoch", "updates", "samples_seen",
                         "train_loss", "validation_loss", "validation_accuracy", "training_seconds"])
        for result in report["results"]:
            for point in result["curve"]:
                writer.writerow([result["optimizer"], result["mode"], result["seed"], point["epoch"],
                                 point["updates"], point["samples_seen"], point["train_loss"],
                                 point["validation"]["loss"], point["validation"]["accuracy"],
                                 point["training_seconds"]])


def plot_results(args, report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib unavailable; metrics.json and curves.csv still contain all curves")
        return None
    fig, axes = plt.subplots(len(args.optimizers), 2, figsize=(11, 4*len(args.optimizers)), squeeze=False)
    colors = dict(sync="black", **{"2bw": "tab:orange", "2bw_ef": "tab:blue"})
    for row, optimizer in enumerate(args.optimizers):
        for result in report["results"]:
            if result["optimizer"] != optimizer:
                continue
            curve = result["curve"]
            label = "{} seed {}".format(result["mode"], result["seed"])
            x = [p["updates"] for p in curve]
            axes[row, 0].plot(x, [p["validation"]["loss"] for p in curve], label=label, color=colors[result["mode"]])
            axes[row, 1].plot(x, [100*p["validation"]["accuracy"] for p in curve], label=label, color=colors[result["mode"]])
        for col, ylabel in enumerate(("Validation cross-entropy", "Validation accuracy (%)")):
            axes[row, col].set(title=optimizer.upper(), xlabel="Completed optimizer updates", ylabel=ylabel)
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend(fontsize=8)
    fig.suptitle("MNIST: matched initialization/data; fresh vs one-update-stale gradients")
    fig.tight_layout()
    path = args.output_dir / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    args, devices, seeds = parse_args(argv)
    configure_determinism(seeds[0])
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  cli={**vars(args), "data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
                  environment={"torch": torch.__version__, "cuda": torch.version.cuda,
                               "devices": [str(d) for d in devices],
                               "device_names": [torch.cuda.get_device_name(d) if d.type == "cuda" else "CPU" for d in devices]},
                  methodology={"training_examples": len(data[2]), "validation_examples": len(data[3]),
                               "test_examples": len(data[1]), "drop_last": True, "split_seed": 1729,
                               "parameter_dtype": "float32", "constant_lr": True, "gradient_clipping": False,
                               "optimizer_placement": "stage device", "profile_is_separate_replay": True,
                               "muon_partition": "hidden block 2D weights: Muon; stem/head/all biases: AdamW; EF applies to both groups",
                               "muon_convention": "B=momentum*B+g; Nesterov g+momentum*B; quintic NS; original shape scaling; unscaled LR for decay",
                               "muon_ns_dtype": args.muon_ns_dtype,
                               "dion3_partition": "hidden block 2D weights: Dion3; stem/head/all biases: AdamW; update-level EF applies to both groups",
                               "dion3_convention": "upstream NorDion2 alias, not original Dion; M+=g; L1 top ceil(fraction*rows); selected residual*=mu; five-iteration Polar Express; selected-row variance EMA; norm-preserving rescale; spectral sqrt(rows/cols) without max clamp",
                               "dion3_ns_dtype": args.dion3_ns_dtype,
                               "dion3_use_triton": args.dion3_triton,
                               "dion3_fallback_betas": [.9, .95], "dion3_epsilon": 1e-8,
                               "dion3_compression_EF": "always active, independent of optional complete-update EF",
                               "dion3_state": "full momentum plus rows-by-1 variance for initialized matrix parameters; AdamW has two full moments and step; optional previous_update is full-sized in both groups",
                               "state_accounting": "actual persistent tensor bytes by state key; excludes transient updates, allocator overhead, gradients and evaluation replica",
                               "profile_seeds": seeds[:1], "epoch_boundary_drains": True,
                               "evaluation_weights": "latest trajectory; separate replica",
                               "training_timing": "includes loading, transfers, updates, epoch drain and any cold kernel JIT/autotuning; excludes evaluation, construction and profiling; mode totals are not a clean speed comparison",
                               "EF": "first completed update ordinary; then x <- x-u-c*(u-u_prev); full uncorrected update stored, moments advanced once",
                               "checkpoint_scope": "optimizer state_dict supported; no full 2BW session checkpoint API",
                               "caveat": "toy convergence example, no claim EF helps all optimizers or reproduces LLM results"},
                  results=[], profiles=[])
    for seed in seeds:
        for name in args.optimizers:
            for mode in args.modes:
                gc.collect()
                result = train_variant(args, devices, seed, name, mode, data)
                report["results"].append(result)
                save_results(args, report)  # Preserve completed runs if later work fails.
                print("TEST {} {} seed={}: loss={:.4f} accuracy={:.2%}".format(
                    name, mode, seed, result["test"]["loss"], result["test"]["accuracy"]), flush=True)
    # One short profile per variant on the first seed, not one trace per epoch.
    if args.profile_steps:
        for name in args.optimizers:
            for mode in args.modes:
                gc.collect()
                report["profiles"].append(profile_variant(args, devices, seeds[0], name, mode, data))
                save_results(args, report)
    plot = plot_results(args, report)
    files = [args.output_dir / name for name in ("metrics.json", "curves.csv")]
    files += [args.output_dir / p["path"] for p in report["profiles"]]
    if plot:
        files.append(plot)
    bundle = args.output_dir / "mnist_convergence_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("bundle integrity failed")
    print("Metrics: {}\nBundle: {}".format(args.output_dir / "metrics.json", bundle), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
