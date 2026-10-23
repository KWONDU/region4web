# Region4Web: Rethinking Observation Space Granularity for Web Agents

Web agents perceive web pages through an observation space, yet its granularity has remained an underexamined design choice. Existing work often represents observation at the same element-level granularity as the action space, leaving the page's functional organization implicit and requiring the agent to infer it from element-level signals at every step. We argue observation should instead operate at the granularity of functional regions, parts of the page whose elements collectively serve a shared purpose. We propose **Region4Web**, a framework that reorganizes the AXTree into functional regions through hierarchical decomposition and semantic abstraction, making the page's functional organization explicit to support page state understanding. Moreover, we propose **PageDigest**, a web-specific inference pipeline that leverages this region-level observation to select task-relevant regions and deliver a compact per-page digest that persists across steps. On the WebArena benchmark, PageDigest substantially reduces observation length across diverse backbone LLMs and established agent methods. Across backbone LLMs, it also improves overall task success. These results show that operating at the granularity of functional regions can provide a more compact and informative basis for page state understanding than element-level processing alone.

## Region4Web

<p align="center"><img src="assets/region4web.png" width="100%"></p>

*Overview of Region4Web inference. Hierarchical decomposition performs a single bottom-up traversal over the page's AXTree, where RegionEncoder computes node representations and EdgeClassifier decides merge or cut for each parent-child edge to determine functional region boundaries. Semantic abstraction then produces each region's purpose and state summary.*

Region4Web is a two-stage framework that constructs region-level observation from a page's AXTree.

### Problem Formulation

At each step, a web agent perceives the current page state through an observation space and selects an action from an action space. The observation can be represented as a tree $\mathcal{T} = (V, E)$, where each node $v \in V$ corresponds to an element on the page with attributes such as role, name, and value. In the prevailing element-level approach, the agent operates over $V$ directly, leaving the page's functional organization implicit in $\mathcal{T}$. Region-level observation makes this organization explicit through a partition $\mathcal{R} = \lbrace R_1, \ldots, R_m \rbrace$ of $V$ into functional regions, where each $R_i$ forms a subtree of $\mathcal{T}$. Each region is associated with a purpose $p_i$ that identifies what the region is for and a state summary $s_i$ that captures its current actionable context. Region4Web learns to produce both $\mathcal{R}$ and the associated $\lbrace (p_i, s_i) \rbrace$ from $\mathcal{T}$.

### Hierarchical Decomposition

Hierarchical decomposition partitions $\mathcal{T}$ into the functional regions in $\mathcal{R}$. We instantiate $\mathcal{T}$ as the page's AXTree, a browser-generated representation that encodes each element's accessibility semantics in a hierarchical structure. Since each $R_i \in \mathcal{R}$ forms a subtree of $\mathcal{T}$, the partition is fully determined by classifying each edge in $E$ as merge or cut. Removing the cut edges from $\mathcal{T}$ splits the tree into subtrees, each of which constitutes a region in $\mathcal{R}$. After the traversal, the subtree remaining connected to the root constitutes the final region in $\mathcal{R}$.

Decomposition learns to predict functional region boundaries from structural cues in the AXTree, while semantic abstraction produces each region's purpose and state summary. Each node $v$ is represented by a feature vector $\mathbf{x}_v$ that combines a learned role embedding with numeric features encoding the node's structural information in $\mathcal{T}$. At each internal node $v$ with children $c_1, \ldots, c_k$ and their respective representations $\mathbf{r}_{c_1}, \ldots, \mathbf{r}_{c_k}$, an EdgeClassifier determines whether each child should be separated, using the sibling mean $\bar{\mathbf{r}} = \frac{1}{k} \sum_j \mathbf{r}_{c_j}$ as context,

$$\hat{y}_{v,c_i} = \mathrm{EdgeClassifier}(\mathbf{x}_v, \mathbf{r}_{c_i}, \bar{\mathbf{r}}).$$

Edges with $\hat{y}_{v,c_i} \geq \tau$ are cut, while the remaining children $\mathcal{M}_v$ are merged into the parent's region. RegionEncoder then computes the parent's representation from $\mathbf{x}_v$ and the merged children $\mathcal{M}_v$,

$$\mathbf{r}_v = \mathrm{RegionEncoder}\left(\mathbf{x}_v, \frac{1}{|\mathcal{M}_v|} \sum_{c_j \in \mathcal{M}_v} \mathbf{r}_{c_j}\right),$$

ensuring that the parent's representation reflects only the children that belong to its region. When $\mathcal{M}_v$ is empty, the aggregation term reduces to 0.

The entire procedure is carried out in a single bottom-up traversal, where each node's representation reflects the boundary decisions already resolved in its subtree, allowing them to propagate upward through the hierarchy without requiring an additional pass.

### Semantic Abstraction

The region partition $\mathcal{R}$ determines which elements belong together, but the semantic meaning of each region remains implicit in its subtree. A fine-tuned language model receives the preprocessed AXTree subtree of each region and produces a purpose $p_i$ and a state summary $s_i$, which address two orthogonal dimensions. Purpose captures what the region is for, serving as the basis for identifying each region. State summary interprets the region's current actionable context, conveying what information and actionable elements are available within it.

### Code and Models

| | Code | Trained model |
|---|---|---|
| Hierarchical decomposition | `framework/decomposition/` | `region4web/decomposition.pt` |
| Semantic abstraction (Qwen3-0.6B) | `framework/abstraction/` | `region4web/abstraction/` (Git LFS) |

## PageDigest

<p align="center"><img src="assets/pagedigest.png" width="70%"></p>

*Overview of PageDigest. At page entry, the actor agent selects task-relevant regions using their abstractions, the task instruction, and action history. Selected regions are exposed as AXTree subtrees, while others retain their purposes, preserving the page's functional organization. Within the same page, PageDigest maintains the per-page digest through observation transition management, while navigation to a new page re-invokes Region4Web.*

Region4Web produces region-level observation for a given page, but deploying it in web environments requires focusing the observation on what is task-relevant and tracking how pages change as the agent acts. PageDigest is a web-specific inference pipeline that constructs a page digest through region selection upon entering a new page and maintains it across steps through observation transition tracking.

### Task-Relevant Region Selection

Upon entering a new page, Region4Web produces the region partition $\mathcal{R}$ and the associated abstractions $\lbrace (p_i, s_i) \rbrace$ for the page. Within PageDigest, the actor agent selects task-relevant regions using these abstractions, together with the task instruction and the action history taken so far. The abstractions specify each region individually and collectively convey the page's overall functional structure and current state, allowing the agent to infer where the task currently stands and what is required next. PageDigest exposes selected regions to the actor agent as their AXTree subtrees with their purposes, preserving element-level granularity for the action space, while non-selected regions are represented by their purposes alone, retaining the page's overall functional organization.

### Page-Aware Observation Transition Management

Within-page observation changes only marginally between consecutive steps. PageDigest therefore tracks observation transitions across steps, rather than reinvoking Region4Web at each step. During transition management, only the region purposes are referenced, since they describe what each region is for and remain stable within the page. In contrast, state summaries capture current actionable context, making them useful for region selection at page entry but less suitable for transition management.

At each step, PageDigest compares the current AXTree with its page-entry state to identify added, removed, and modified nodes. In the page-entry AXTree, removed nodes are deleted and modified nodes' values updated under existing region purposes. Added nodes are listed separately while retaining their structural grouping, since merging them into existing regions could shift those regions' purposes. Upon a URL change, Region4Web produces the new page's $\mathcal{R}$ and $\lbrace (p_i, s_i) \rbrace$, and region selection proceeds.

PageDigest shares the actor agent's backbone LLM and operates solely on the observation space, requiring no additional model or modification to the actor agent's policy and making it directly applicable to diverse web agents. Moreover, since region selection depends on the actor agent's backbone capability, the agent is given an additional `view_all` action that reveals all regions as full AXTree subtrees for the remainder of the page, providing a fallback when the selected regions are insufficient.

### Code

Region selection is implemented in `pipeline/selection/`, observation transition management in `pipeline/transition.py` and `pipeline/manager.py`, and the actor-facing observation in `pipeline/observation_builder.py`.

## Setup

```bash
git lfs install && git lfs pull        # fetch region4web/abstraction/model.safetensors
pip install -e .
pip install -r requirements.txt
playwright install chromium
cp secrets.yaml.example secrets.yaml   # API keys for the backbone and annotator LLMs
```

Set your WebArena URLs in `config.yaml` under `benchmark.webarena.urls`. Map tasks use the live OpenStreetMap service by default.

## Inference

The runner evaluates WebArena tasks in one of two observation modes.

| `--mode` | Observation given to the actor |
|---|---|
| `backbone` | Full AXTree at every step (baseline) |
| `pagedigest` | Region4Web + PageDigest |

```bash
# PageDigest with GPT-5.1 on all 812 WebArena tasks
python -m benchmarks.runner --backbone gpt-5.1 --mode pagedigest \
    --task-list benchmarks/task_lists/full.txt

# Backbone-only baseline
python -m benchmarks.runner --backbone gpt-5.1 --mode backbone \
    --task-list benchmarks/task_lists/full.txt

# One runner per site in parallel (shopping, shopping_admin, reddit, gitlab, map)
python -m benchmarks.run_parallel --backbone gpt-5.1 --mode pagedigest \
    --task-list benchmarks/task_lists/full.txt
```

- `benchmarks/task_lists/lite.txt` runs the 165-task WebArena-Lite subset, and `--task-slice 0:10` runs part of a list.
- `--backbone <name>` merges `configs/<name>.yaml` into `config.yaml`. Model names with the `openai/`, `anthropic/`, `deepseek/`, or `gemini/` prefix call the provider API, and other names (e.g., `Qwen/Qwen3.5-27B`) call an OpenAI-compatible vLLM server at `benchmark.llm.base_url`.
- With `run_parallel` in `pagedigest` mode, the number of parallel runners is capped at the number of visible GPUs, since each runner loads the Region4Web models.

Each task writes to `results/webarena/<backbone>/base/<mode>/<task_id>/`:

| File | Content |
|---|---|
| `summary.json` | Success, final answer, steps, token usage and cost |
| `trajectory.jsonl` | Per-step action and timing |
| `axtrees.jsonl` | Serialized AXTree per step |
| `decompositions.jsonl`, `abstractions.jsonl`, `selections.jsonl` | Region4Web and selection outputs per page entry |
| `transitions.jsonl` | Within-page observation changes |
| `actor_calls.jsonl` | Full actor prompt and output per step |

## Training

### Data annotation

Training data is annotated by gpt-5-mini through the OpenAI Batch API. The source pages' AXTrees are read from `data/axtrees/`. Each `generate_*` script writes a batch request file to `data/api_requests/`. Submit it with `batch_api.py upload` and fetch the results with `batch_api.py download` before running the next script.

```bash
# 1. Decomposition: region root IDs per page, then re-partition oversized regions (fallback)
python training/data_generation/generate_decomposition_data.py --mode batch
python training/data_generation/generate_decomposition_fallback.py
python training/data_generation/build_decomposition_annotations.py \
    --fallback-output data/api_requests/decomposition_fallback_output.jsonl

# 2. Verification: keep pages whose partition has no invalid region
python training/data_generation/generate_verification_data.py
python training/data_generation/build_verification_annotations.py

# 3. Abstraction: purpose and state summary per verified region
python training/data_generation/generate_abstraction_data.py
python training/data_generation/build_abstraction_annotations.py

# 4. Final training files: data/decomposition.jsonl, data/abstraction.jsonl
python training/data_generation/build_training_data.py
```

```bash
# Batch API helper (split / upload / status / download / cancel)
python training/data_generation/batch_api.py upload   --input data/api_requests/decomposition_input.jsonl
python training/data_generation/batch_api.py download --input data/api_requests/decomposition_input.jsonl
```

### Model training

```bash
# Decomposition model: teacher forcing on cut/merge edge labels, focal loss (α = 0.75, γ = 2.0)
python -m training.train_decomposition --config config.yaml

# Abstraction model: full fine-tuning of Qwen3-0.6B with DDP (effective batch size 48)
torchrun --nproc_per_node=3 training/train_abstraction.py --config config.yaml
```

Checkpoints are written to `checkpoints/`. Hyperparameters are set under `training` in `config.yaml`. To use a newly trained model, point `framework.decomposition.checkpoint_path` or `framework.abstraction.checkpoint_path` to it.
