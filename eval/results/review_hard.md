# Answer review

For each item replace the `?` after VERDICT with:
2 = correct and supported by the passages, 1 = partly correct or partly unsupported, 0 = wrong or not supported by the passages.

## h12
**Question:** can I run several reward functions that call web services at the same time?

**Answer:**
> Yes. If you define your reward functions as `async def` coroutines, the trainer will await them concurrently (via `asyncio.gather`), so multiple web‑service calls can run in parallel and their latencies overlap. [1][3]

**Passages cited**:
### [1] trl/grpo_trainer.md | GRPO Trainer > Customization > Using a custom reward function > Example 5: Asynchronous reward functions
```text
Custom reward functions can also be defined as `async def` coroutines. This is useful if your reward depends on slow I/O (for example, calling a remote service). When you pass multiple async reward functions, [`GRPOTrainer`] executes them concurrently so their latency overlaps.

Below is a minimal example of an async reward function that simulates an I/O-bound operation:

```python
import asyncio

async def async_reward_func(prompts, completions, **kwargs):
    # Simulate an I/O-bound call (e.g., HTTP request, database lookup)
    await asyncio.sleep(0.01)
    # Simple toy reward: 1.0 if the completion is non-empty, else 0.0
    return [1.0 if completion else 0.0 for completion in completions]
```
```
### [3] trl/grpo_trainer.md | GRPO Trainer > Customization > Using a custom reward function
```text
Reward functions can be either synchronous Python callables or asynchronous `async def` coroutines. When you provide multiple asynchronous reward functions, they are awaited concurrently (run in parallel via `asyncio.gather`) so their latency overlaps.

1. **Input arguments**:
   - The function must accept the following as keyword arguments:
     - `prompts` (contains the prompts),
     - `completions` (contains the generated completions),
     - `completion_ids` (contains the tokenized completions),
     - `trainer_state` ([`~transformers.TrainerState`]): The current state of the trainer. This can be used to implement dynamic reward functions, such as curriculum learning, where the reward is adjusted based on the training progress.
     - `log_extra`: a callable `log_extra(column: str, values: list)` to add extra columns to the completions table. See Example 6.
     - `log_metric`: a callable `log_metric(name: str, value: float)` to log scalar metrics as plots alongside `kl`, `entropy`, etc. See Example 6.
     - `environments`: a list of environment instances, one per completion. Only present when `environment_factory` is provided. Use this to read state accumulated during the ep
```

VERDICT: ?
NOTES:

---

## h03
**Question:** max_inflight_tasks

**Answer:**
> `max_inflight_tasks` is a concurrency limit used by the AsyncGRPO trainer.  
> * It caps the number of requests that can be sent to the vLLM server at once.  
> * By default it is computed automatically as  
> 
> \[
> \text{max\_staleness} \times \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_processes}
> \]
> 
> which is the maximum number of samples the trainer can consume before they become stale.  
> * The value is a ceiling; the actual in‑flight count may be lower.  
> * In OpenEnv integration it also controls how many rollouts run concurrently, and the server’s `MAX_CONCURRENT_ENVS` must be at least `max_inflight_tasks + 1` [3].  
> 
> [1][4]

**Passages cited**:
### [1] trl/async_grpo_trainer.md | Asynchronous GRPO > How it differs from [`GRPOTrainer`] > Concurrency
```text
The number of concurrent requests sent to the vLLM server is capped by `max_inflight_tasks`. By default it is set automatically to `max_staleness × per_device_train_batch_size × gradient_accumulation_steps × num_processes` — the maximum number of samples the trainer can consume before they become stale. Generating more than this is wasteful since the excess samples will be discarded.

`max_inflight_tasks` is a ceiling, not the number actually in flight. The worker polls the server's `/metrics` endpoint every few seconds and keeps a live cap underneath it (`rollout/inflight_cap`): past a certain concurrency, more requests make vLLM *slower*, because the KV cache fills, sequences are preempted and recomputed, and cached prefixes are evicted and re-prefilled. The cap starts at `KV cache tokens / max_model_len`, read from the server at startup: as many requests as fit at full length, so the first step cannot thrash. While usage stays under 0.6 with nothing queued, the cap grows by 25% each time a cap's worth of rollouts has completed, up to the ceiling. When KV cache usage passes 0.8, the cap is cut by 30%; no running request is cancelled, admissions just stop until the pool has draine
```
### [3] trl/openenv.md | OpenEnv Integration for Training LLMs with Environments > Training on harnesses: training real coding agents (Harbor) > How it works
```text
Each rollout runs in its own isolated session, with its own sandbox and its own capture session. The isolation matters for two reasons: the proxy has to capture exactly that rollout's tokens, and one rollout must not interfere with another. The `max_inflight_tasks` setting controls how many rollouts run at the same time, and the server's `MAX_CONCURRENT_ENVS` must be at least `max_inflight_tasks + 1` (the factory keeps one extra connection for task metadata).
```
### [4] trl/async_grpo_trainer.md | Asynchronous GRPO > How it differs from [`GRPOTrainer`] > Concurrency
```text
The number of concurrent requests sent to the vLLM server is capped by `max_inflight_tasks`. By default it is set automatically to `max_staleness × per_device_train_batch_size × gradient_accumulation_steps × num_processes` — the maximum number of samples the trainer can consume before they become stale. Generating more than this is wasteful since the excess samples will be discarded.
```

VERDICT: ?
NOTES:

---

## h20
**Question:** which logged metric shows how random my policy's choices are in online DPO?

**Answer:**
> The logged metric that indicates how random your policy’s choices are in online DPO is **`objective/entropy`** – the mean entropy of the model, which reflects the randomness of the actions chosen by the policy. [1]

**Passages cited**:
### [1] trl/online_dpo_trainer.md | Online DPO Trainer > Logged metrics
```text
While training and evaluating, we record the following metrics. Here is an example [tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/w4apmsi9)

* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current model and reference model.
* `objective/entropy`: The mean entropy of the model, indicating the randomness of the actions chosen by the model.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `scores - non_score_reward`. The `rlhf_reward` is the ultimate objective of online DPO training. If training works as intended, this metric should keep going up.
* `objective/scores`: The mean scores returned by the reward model.
* `objective/scores_margin`: The mean score margin (according to the external reward model) between the chosen and rejected completions.
* `rewards/chosen`: The mean reward (according to online DPO's implicit reward model)of the chosen completions.
* `rewards/rejected`: The mean reward (according to online DPO's implicit rewa
```

VERDICT: ?
NOTES:

---

## h19
**Question:** what do I do when torch.export chokes on a pattern in my model during export?

**Answer:**
> If `torch.export` cannot trace a part of your model, add a **patch** that replaces the problematic code only for the export run.
> 
> 1. Write a factory that returns a replacement (e.g., a no‑op or a simpler implementation).  
> 2. Register that factory with `register_patch`, specifying the exporter stage (e.g., `"dynamo"`) and the dotted path of the method or op that needs to be patched.  
> 3. The exporter will swap the method in before tracing and restore it afterward, so the patch only affects the export.
> 
> Example (from the docs):
> 
> ```python
> from transformers.exporters.utils import register_patch
> 
> @register_patch("dynamo", "transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier")
> def _patch_classifier_cast(_original):
>     # Replace the untraceable dtype cast with a no‑op during export.
>     return lambda self, *args, **kwargs: None
> ```
> 
> This mechanism can also be used to patch a `torch` op or to write a graph‑rewrite “fix” instead of a patch.

**Passages cited** (none cited, showing top 2):
### [1] transformers/exporters_extend.md | Extending the exporters
```text
`torch.export` traces the model into a graph, later stages transform
that graph, and a final stage lowers or emits it for the target runtime. Most models pass through
untouched. When there's a PyTorch pattern the backend can't handle, the exporter applies a
small workaround at that stage rather than editing the model.

Add a workaround by writing one function and registering it with a decorator. Each workaround belongs at the lowest stage that can express it cleanly.
```
### [2] transformers/exporters.md | Exporters > Limitations and workarounds
```text
`torch.export`, `torch.onnx.export`, and ExecuTorch each have rough edges around specific
PyTorch patterns. The exporters work around these with a small set of reversible patches
and FX-level fixes applied at well-defined points in the export flow. None of this is
visible from the public `export` API, but the most common things to know:

- FlashAttention and FlexAttention are not exportable on any backend. `sdpa` is the preferred
setting and `eager` also works (slower). Set one of them on the model before calling `export`
if it's using something else.
- `grouped_mm` traces fine through `DynamoExporter` and is auto-translated for `OnnxExporter`.
For `ExecutorchExporter` with either XNNPACK or MLX, the exporter swaps MoE experts to
`batched_mm` before export.
```

VERDICT: ?
NOTES:

---

## h08
**Question:** Accelerator.get_tracker wandb

**Answer:**
> Use `Accelerator.get_tracker` with the tracker’s name:
> 
> ```python
> wandb_tracker = accelerator.get_tracker("wandb")
> ```
> 
> You can then interact with the underlying `wandb` run object, e.g. `wandb_tracker.log_artifact(...)` [1].

**Passages cited**:
### [1] accelerate/usage_guides/tracking.md | Experiment trackers > Accessing the internal tracker
```text
If some custom interactions with a tracker might be wanted directly, you can quickly access one using the
[`Accelerator.get_tracker`] method. Just pass in the string corresponding to a tracker's `.name` attribute
and it will return that tracker on the main process.

This example shows doing so with wandb:

```python
wandb_tracker = accelerator.get_tracker("wandb")
```

From there you can interact with `wandb`'s `run` object like normal:

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

Trackers built in Accelerate will automatically execute on the correct process,
  so if a tracker is only meant to be ran on the main process it will do so
  automatically.

If you want to truly remove Accelerate's wrapping entirely, you can
achieve the same outcome with:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
if accelerator.is_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```
```

VERDICT: ?
NOTES:

---

## h07
**Question:** _supports_assign_param_buffer

**Answer:**
> `_supports_assign_param_buffer` is a flag that custom models can set to indicate whether the super‑fast initialization routine can be applied to that model. If `test_save_and_load_from_pretrained` fails, you should set this flag to `False` so that the super‑fast init is skipped. [1]

**Passages cited**:
### [1] transformers/main_classes/model.md | Models > PreTrainedModel
```text
- push_to_hub
    - all

Custom models should also include a `_supports_assign_param_buffer`, which determines if superfast init can apply
on the particular model. Signs that your model needs this are if `test_save_and_load_from_pretrained` fails. If so,
set this to `False`.
```

VERDICT: ?
NOTES:

---

## h18
**Question:** what is the slowest but most accurate way to quantize to 4 bits with AutoRound?

**Answer:**
> The slowest but most accurate way to quantize to 4‑bit with AutoRound is to use the **AutoRoundBest** recipe.  
> It gives the best accuracy in most scenarios (though it is 4–5× slower than the standard AutoRound) and is recommended for 2‑bit quantization, but it can also be used for 4‑bit settings.
> 
> ```python
> from auto_round import AutoRound
> 
> model_name = "Qwen/Qwen3-0.6B"
> ar = AutoRound(
>     model_name,
>     scheme="W4A16",   # 4‑bit weights, 16‑bit activations
>     nsamples=512,
>     iters=1000,
> )
> 
> output_dir = "./tmp_autoround"
> ar.quantize_and_save(output_dir, format='auto_round')
> ```
> 
> [1]

**Passages cited**:
### [1] transformers/quantization/auto_round.md | AutoRound > Quantization and Serialization (offline) > AutoRoundBest recipe
```text
This setting provides the best accuracy in most scenarios but is 4–5× slower than the standard AutoRound recipe. It is especially recommended for 2-bit quantization and is a good choice if sufficient resources are available.

```python
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
ar = AutoRound(
    model_name,
    scheme="W4A16",
    nsamples=512,
    iters=1000,
)

output_dir = "./tmp_autoround"
ar.quantize_and_save(output_dir, format='auto_round')
```
```

VERDICT: ?
NOTES:

---

## h04
**Question:** init_empty_weights

**Answer:**
> `init_empty_weights` is a context manager from **Accelerate** that lets you instantiate a model without allocating any RAM.  
> Inside the context, each parameter is created on PyTorch’s **meta** device, so the model is only a “skeleton” and no weight data is loaded until you later load a checkpoint or quantize it.
> 
> ```py
> from accelerate import init_empty_weights
> from transformers import AutoConfig, AutoModelForCausalLM
> 
> config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
> with init_empty_weights():
>     model = AutoModelForCausalLM.from_config(config)   # no RAM used yet
> ```
> 
> This technique is used for very large models (e.g., Mixtral‑8x7B) and is the first step in Accelerate’s Big Model Inference workflow. It relies on PyTorch 1.9’s meta device and cannot be moved to CPU or another device until weights are loaded. [1][2]

**Passages cited**:
### [1] accelerate/quicktour.md | Quicktour > Big Model Inference > Empty weights initialization
```text
The [`~accelerate.init_empty_weights`] context manager initializes models of any size by creating a *model skeleton* and moving and placing parameters each time they're created to PyTorch's [**meta**](https://pytorch.org/docs/main/meta.html) device. This way, not all weights are immediately loaded and only a small part of the model is loaded into memory at a time.

For example, loading an empty [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model takes significantly less memory than fully loading the models and weights on the CPU.

```py
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```
```
### [2] accelerate/concept_guides/big_model_inference.md | Loading big models into memory > How the Process Works: Working with Code > Instantiating an empty model
```text
The first tool Accelerate introduces to help with big models is a context manager [`init_empty_weights`] that helps you initialize a model without using any RAM so that step 1 can be done on models of any size. Here is how it works:

```py
from accelerate import init_empty_weights

with init_empty_weights():
    my_model = ModelClass(...)
```

For instance:

```py
with init_empty_weights():
    model = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
```

initializes an empty model with a bit more than 100B parameters. Behind the scenes, this relies on the meta device introduced in PyTorch 1.9. During the initialization under the context manager, each time a parameter is created, it is instantly moved to that device.

You can't move a model initialized like this on CPU or another device directly, since it doesn't have any data. It's also very likely that a forward pass with that empty model will fail, as not all operations are supported on the meta device.
```

VERDICT: ?
NOTES:

---

## h01
**Question:** TRF058

**Answer:**
> **TRF058** – *Buffer registration rule*  
> In `modeling_*.py` and `modular_*.py`, any call to `register_buffer("<name>", …)` whose buffer name is a string literal is flagged.  
> - Computed names (variables or f‑strings) are exempt.  
> - With `torch>=2.5`, `nn.Buffer` registers a buffer via plain attribute assignment (`self.foo = nn.Buffer(...)`).  
> - Buffers created only by a method call exist only as a side‑effect of `__init__`; to modify such a buffer a modular file must redefine the entire `__init__`.  
> - Assigning a buffer as an attribute allows inheritance and overriding.  
> 
> [1]

**Passages cited**:
### [1] transformers/modeling_rules.md | Model structure rules > Rules reference > TRF058
```text
In modeling_*.py and modular_*.py, flags `register_buffer("<name>", ...)` calls whose buffer name is a string literal, on any receiver (`self`, or another module such as `layer.mamba`). A computed name -- a variable or f-string, e.g. one buffer per layer inside a loop -- has no attribute-assignment equivalent and is exempt. Since torch>=2.5 `nn.Buffer` registers a buffer through plain attribute assignment, like `nn.Parameter`. A buffer created by a method call only exists as a side effect of running `__init__`, so a modular file that wants to tweak one has to redefine the whole `__init__`. Assigned as an attribute, it can be inherited and overridden on its own.

```diff
-        self.register_buffer("inv_freq", inv_freq, persistent=False)
-        self.register_buffer(
-            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
-        )
+        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
+        self.position_ids = nn.Buffer(torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
```
```

VERDICT: ?
NOTES:

---

## h10
**Question:** I stored audio in a dataset but I want the file path, not decoded arrays

**Answer:**
> Use the `Audio` feature with `decode=False` so that the column returns the file path (or bytes) instead of a decoded `AudioDecoder` object.
> 
> ```python
> from datasets import load_dataset, Audio
> 
> # Load the dataset and cast the audio column to not decode
> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")\
>           .cast_column("audio", Audio(decode=False))
> 
> # Now each example contains the path
> print(dataset[0]["audio"])
> # {'bytes': None, 'path': '/root/.cache/huggingface/.../602ba55abb1e6d0fbce92065.wav'}
> ```
> 
> Alternatively, for an existing dataset you can disable decoding globally:
> 
> ```python
> dataset = dataset.decode(False)   # returns paths/bytes instead of decoded arrays
> ```
> 
> [1][2]

**Passages cited**:
### [1] datasets/about_dataset_features.md | Dataset features > Audio feature
```text
Audio datasets have a column with type [`Audio`], which contains three important fields:

- `array`: the decoded audio data represented as a 1-dimensional array.
- `path`: the path to the downloaded audio file.
- `sampling_rate`: the sampling rate of the audio data.

When you load an audio dataset and call the audio column, the [`Audio`] feature automatically decodes and resamples the audio file:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset[0]["audio"]
<datasets.features._torchcodec.AudioDecoder object at 0x11642b6a0>
```

> [!WARNING]
> Index into an audio dataset using the row index first and then the `audio` column - `dataset[0]["audio"]` - to avoid decoding and resampling all the audio files in the dataset. Otherwise, this can be a slow and time-consuming process if you have a large dataset.

With `decode=False`, the [`Audio`] type simply gives you the path or the bytes of the audio file, without decoding it into an torchcodec `AudioDecoder` object,
```
### [2] datasets/about_dataset_features.md | Dataset features > Audio feature
```text
With `decode=False`, the [`Audio`] type simply gives you the path or the bytes of the audio file, without decoding it into an torchcodec `AudioDecoder` object,

```py
>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train").cast_column("audio", Audio(decode=False))
>>> dataset[0]
{'audio': {'bytes': None,
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav'},
 'english_transcription': 'I would like to set up a joint account with my partner',
 'intent_class': 11,
 'lang_id': 4,
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'transcription': 'I would like to set up a joint account with my partner'}
```
```

VERDICT: ?
NOTES:

---
