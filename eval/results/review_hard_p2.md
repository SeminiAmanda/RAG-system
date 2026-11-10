# Answer review

For each item replace the `?` after VERDICT with:
2 = correct and supported by the passages, 1 = partly correct or partly unsupported, 0 = wrong or not supported by the passages.

## h18
**Question:** what is the slowest but most accurate way to quantize to 4 bits with AutoRound?

**Answer:**
> The slowest but most accurate way to quantize a model to 4‑bit precision with AutoRound is to use the **AutoRoundBest** recipe, which prioritizes accuracy over speed and is 4–5× slower than the standard AutoRound recipe. This setting is recommended when you have sufficient resources and want the best possible accuracy, even for 4‑bit quantization. [1]

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

## h16
**Question:** how do I talk to my local transformers server using the Hugging Face client?

**Answer:**
> To use the local **transformers serve** instance from a Hugging Face client, point the client at the OpenAI‑compatible API that the server exposes.  
> The server runs on `http://localhost:8000` and exposes a `/v1` prefix. For example, you can list the models that the server has loaded by querying:
> 
> ```bash
> curl http://localhost:8000/v1/models
> ```
> 
> or by using the Hugging Face client to send a request to that URL.  
> If you are using a web UI such as Open WebUI, set the **Speech‑to‑Text Engine** to **OpenAI** and the URL to `http://localhost:8000/v1` (and enable CORS if needed) so that the UI talks to your local server.  
> 
> These steps let the client communicate with the local transformers server.  
> [2][4]

**Passages cited**:
### [2] transformers/serve-cli/serving.md | Serve CLI > v1/models
```text
The `v1/models` endpoint scans your local Hugging Face cache and returns a list of downloaded models in the OpenAI-compatible format. Third-party tools use this endpoint to discover available models.

Download a model before running `transformers serve`.

```bash
transformers download Qwen/Qwen2.5-0.5B-Instruct
```

Once downloaded, the model appears in `/v1/models` responses.

```shell
curl http://localhost:8000/v1/models
```

The endpoint returns a JSON object with available models.
```
### [4] transformers/serve-cli/openweb_ui.md | Open WebUI
```text
[Open WebUI](https://openwebui.com/) provides a self-hosted interface for offline models. Enable [CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS) to serve local models (it's disabled by default for security).

```bash
transformers serve --enable-cors
```

To use the server for speech-to-text, go to **Settings > Audio**.

1. Set **Speech-to-Text Engine** to **OpenAI**.
2. Set the URL to http://localhost:8000/v1.
3. Enter a transcription model in **SST Model** (openai/whisper-large-v3). Find more models on the [Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending).

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_openwebui_stt_settings.png"/>
</h3>

Start a chat and speak. The model transcribes your audio into the input field.
```

VERDICT: ?
NOTES:

---

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

## h19
**Question:** what do I do when torch.export chokes on a pattern in my model during export?

**Answer:**
> If `torch.export` fails to trace a particular PyTorch pattern in your model, you can add a small workaround by writing a patch function and registering it with the exporter.  
> Write a factory that returns a replacement (e.g., a no‑op or a simpler implementation) and decorate it with `@register_patch`, pointing to the dotted path of the problematic method or op. The exporter will swap the method in before tracing and restore it afterward, so the patch only affects the export process. This is the recommended way to handle patterns that `torch.export` cannot trace directly. [1][4]

**Passages cited**:
### [1] transformers/exporters_extend.md | Extending the exporters
```text
`torch.export` traces the model into a graph, later stages transform
that graph, and a final stage lowers or emits it for the target runtime. Most models pass through
untouched. When there's a PyTorch pattern the backend can't handle, the exporter applies a
small workaround at that stage rather than editing the model.

Add a workaround by writing one function and registering it with a decorator. Each workaround belongs at the lowest stage that can express it cleanly.
```
### [4] transformers/exporters_extend.md | Extending the exporters > Add a patch
```text
Suppose a model method does something `torch.export` can't trace. NLLB-MoE's
`NllbMoeTop2Router._cast_classifier` casts the classifier weights to another dtype,
which isn't traceable. Replace it with a no-op for the duration of the export.

Write a factory that takes the original method and returns its replacement, then register the
factory against the method's dotted path:

```python
from transformers.exporters.utils import register_patch

@register_patch("dynamo", "transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier")
def _patch_classifier_cast(_original):
    # Replace the untraceable dtype cast with a no-op during export.
    return lambda self, *args, **kwargs: None
```

The exporter swaps the method in before tracing and restores it afterward,
so the patch only affects export. A few variations:

- Pass extra paths to share one factory across call sites, for example
  `@register_patch("dynamo", path_a, path_b)`.
- Patch a `torch` op by pointing the path at it, for example `@register_patch("onnx", "torch.where")`.
  The factory receives the real op as its argument, so the replacement can call through to it.
- Write a fix instead of a patch when you
```

VERDICT: ?
NOTES:

---

## h08
**Question:** Accelerator.get_tracker wandb

**Answer:**
> Use `Accelerator.get_tracker` to retrieve the WandB tracker instance:
> 
> ```python
> wandb_tracker = accelerator.get_tracker("wandb")
> ```
> 
> You can then interact with it as you would normally with WandB, e.g.:
> 
> ```python
> wandb_tracker.log_artifact(some_artifact_to_log)
> ```
> 
> If you want to bypass Accelerate’s automatic process handling, pass `unwrap=True` and manually check `accelerator.is_main_process` before logging:
> 
> ```python
> wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
> if accelerator.is_main_process:
>     wandb_tracker.log_artifact(some_artifact_to_log)
> ```
> 
> [1]

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
> `_supports_assign_param_buffer` is a flag that custom models can expose to indicate whether the “superfast init” mechanism can be applied to that model. If a model’s `test_save_and_load_from_pretrained` fails, you should set this flag to `False` so that superfast init is disabled for that model. [1]

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

## h20
**Question:** which logged metric shows how random my policy's choices are in online DPO?

**Answer:**
> The logged metric that indicates how random your policy’s choices are in online DPO is **`objective/entropy`** – it reports the mean entropy of the model, which reflects the randomness of the actions chosen by the policy. [1]

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

## h04
**Question:** init_empty_weights

**Answer:**
> `init_empty_weights` is a context manager from the Accelerate library that lets you instantiate a model without allocating any real memory for its parameters.  
> Inside the context, each parameter is created on PyTorch’s **meta** device, so the model is only a “skeleton” and no weights are loaded into RAM until you later load them or quantize the model. This makes it possible to start the first step of building very large models that would otherwise not fit in memory.  
> Because the parameters are on the meta device, the model cannot be moved to CPU or another device directly, and a forward pass will usually fail until the weights are loaded or quantized.  
> The context manager is also used in the quantization workflow, where an empty model is first created with `init_empty_weights` before loading and quantizing the weights.  
> 
> [1][2][4]

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
### [4] accelerate/usage_guides/quantization.md | Model quantization > `bitsandbytes` Integration > How it works
```text
First, we need to initialize our model. To save memory, we can initialize an empty model using the context manager [`init_empty_weights`].

Let's take the GPT2 model from minGPT library.

```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    empty_model = GPT(model_config)
```

Then, we need to get the path to the weights of your model. The path can be the state_dict file (e.g. "pytorch_model.bin") or a folder containing the sharded checkpoints.

```py
from huggingface_hub import snapshot_download
weights_location = snapshot_download(repo_id="marcsun13/gpt2-xl-linear-sharded")
```

Finally, you need to set your quantization configuration with [`~utils.BnbQuantizationConfig`].

Here's an example for 8-bit quantization:

```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
```

Here's an example for 4-bit quantization:

```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config
```

VERDICT: ?
NOTES:

---

## h15
**Question:** how do I lay out 4 accelerators as a grid for splitting a model at inference?

**Answer:**
> Use Hugging Face’s `DeviceMesh` to create a grid of the four accelerators.  
> For a simple 1‑D layout you can do:
> 
> ```python
> import torch
> from torch.distributed.device_mesh import init_device_mesh
> 
> device_type = torch.accelerator.current_accelerator().type
> device_mesh = init_device_mesh(device_type, (4,), mesh_dim_names=["tp"])
> ```
> 
> This creates a 1‑D mesh of four devices that can be used for tensor‑parallel inference.  
> If you need a multi‑dimensional grid, pass a shape tuple with more than one dimension to `init_device_mesh`. [3]

**Passages cited**:
### [3] transformers/perf_infer_gpu_multi.md | Tensor parallelism for inference > Design implementation > DeviceMesh
```text
`DeviceMesh` creates a multi-dimensional grid of devices that communicate together. Different parallelization strategies require different communication patterns. Create a `DeviceMesh` with multiple sub-meshes to handle these patterns.

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

# Create a 1D mesh of 4 accelerators
device_type = torch.accelerator.current_accelerator().type
device_mesh = init_device_mesh(device_type, (4,), mesh_dim_names=["tp"])
```

Most `torch.distributed` parallelization strategies apply to the mesh itself or its sub-mesh. The mesh automatically handles communication patterns.
```

VERDICT: ?
NOTES:

---

## h01
**Question:** TRF058

**Answer:**
> TRF058 is a rule that governs how buffers should be registered in the Hugging Face transformers codebase.  
> It requires that calls to `register_buffer("<name>", …)` use a string literal for the buffer name on any module (`self` or a sub‑module such as `layer.mamba`). Buffers whose names are computed (e.g., via a variable or f‑string inside a loop) are exempt because they have no attribute‑assignment equivalent. Since `torch>=2.5`, `nn.Buffer` registers a buffer through plain attribute assignment, so the recommended pattern is to assign the buffer as an attribute (e.g., `self.inv_freq = nn.Buffer(...)`). A buffer created only by a method call exists only as a side effect of `__init__`; to modify it a modular file must redefine the entire `__init__`. Assigned as an attribute, the buffer can be inherited and overridden independently. [1]

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
