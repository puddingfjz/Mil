# Data and model preparation

[简体中文](data_and_models.zh-CN.md)

The scheduler and inference entry resolve model IDs to the same pinned local
snapshots checked by preflight; execution never implicitly downloads `main`.
The scheduling entry point uses `MIL_DATA_ROOT` and repository-relative paths
for datasets and cost resources. The doctor checks active code for hard-coded
machine-specific paths.

## Model policy

- Model weights are never committed to this repository.
- Every model must be pinned to an immutable Hugging Face revision.
- The pinned model card/license evidence, access requirement, and known legal
  caveats are recorded per model; model weights remain evaluator-supplied.
- A local cache may be used, but the resolved snapshot revision is recorded in
  run metadata.

`MIL_MODEL_ROOT` selects the Hugging Face cache root; otherwise `HF_HUB_CACHE`,
`HF_HOME/hub`, or the default Hugging Face cache is used, in that order.
Prepare the snapshots before execution. A missing cache is an error, not a
warning. Preflight returns each model's revision, absolute snapshot path and
configuration checksum. The runner freezes this mapping in run/task metadata
and passes it to child processes through `MIL_MODEL_SNAPSHOTS`; scheduling
configuration reads, request tokenization, runtime prompt tokenizers and inference
weights all use it. Task processes run with Hugging
Face/Transformers offline mode enabled. A missing, changed or unselected snapshot
fails instead of falling back to another cache or revision.
This includes the engine configuration used while generating initial execution
plans. Only the explicit CPU-only scheduling route retains model IDs for its
frozen configuration reader; it does not require a model cache. The normal
experiment runner clears that CPU-only mode before starting a runtime task.

Preflight checks only the resources needed by the expanded selection of tasks
(`--suite`, `--method`, `--parameter`). Router checks its selected question
partition and five routing models; Ensembling checks its model set (or the
selected Oracle subset); Chain Summary checks both book inputs and the selected
summary/evaluator models; Mixed takes their union. All tasks retain the bundled
output-length inputs. Missing selected resources still fail. Model/data disk
capacity is estimated from the selected payloads plus bootstrap overhead and
runtime reserve, rather than the full inventory. Resource preparation commands
may still verify the full inventory unless given their explicit resource filters.

`manifests/models.yaml` preserves fixed revisions, snapshot checksums and
access evidence. Revision provenance cites source documents at fixed Git
commits. These references do not establish final-paper membership for every
historical run.

The pinned model inventory has the following access requirements:

| Models | Declared terms in pinned evidence | Required handling |
| --- | --- | --- |
| Llama-2-70B, CodeLlama-34B | Llama 2 | Complete Meta's gated contact/license flow |
| Mixtral-8x7B, Mistral-7B | Apache-2.0 | Pinned cards contain Hugging Face gated-contact metadata; recheck the current host gate |
| WizardLM-13B, Vicuna-13B | Llama 2 | Satisfy the upstream and Llama 2 terms |
| OASST Pythia-12B | Apache-2.0 | Satisfy upstream terms |
| Alpaca-13B | Undeclared | Do not fetch/use for release until its LLaMA-derived weight/data terms are resolved |
| Baize-v2-13B | CC-BY-NC-4.0 | Noncommercial and underlying LLaMA-term review |
| Koala-13B | Custom/composite, academic-research-only | No commercial use; review cited LLaMA, OpenAI-data, and ShareGPT terms |
| Dolly-v2-12B | MIT | Satisfy upstream terms; its card separately identifies training data as CC-BY-SA |
| MPT-7B-chat | CC-BY-NC-SA-4.0 | Noncommercial use only |
| ChatGLM3-6B | Custom ChatGLM3 terms | Commercial use requires registration |
| StableLM tuned alpha | CC-BY-NC-SA-4.0 | Noncommercial use only |

This table summarizes the exact evidence hashes in `manifests/models.yaml`; it
does not broaden any upstream grant. Hosting gates can change independently of
a Git revision, so the preparation script never claims that cached access
proves a new evaluator is authorized.

Each manifest entry also freezes `config.json` SHA-256, snapshot/weight file
counts and byte totals, and a sorted weight path/size digest. The preparation
and doctor commands compare all of them and hash the listed card/license files.
The lightweight weight-layout digest detects absent or truncated shards without
rereading approximately 0.9 TB on every preflight. It is explicitly not a
weight-content digest: the immutable repository revision remains the upstream
content identifier.

## Dataset policy

- `MIL_DATA_ROOT` identifies the external dataset root.
- Repository-owned prepared inputs use `${MIL_REPO_ROOT}`.
- Every published prepared dataset gets a SHA-256 checksum, sample count,
  preprocessing command, upstream URL, and license.
- Existing output-length files are treated as versioned inputs when used to
  force previously observed lengths; newly generated files belong to a run
  directory.

The dataset manifest separates byte/semantic provenance from redistribution
permission. The following source checks were completed on 2026-09-06:

- MixInstruct is pinned to revision
  `c7f77a4ef0515a99d1752c6387c22366d6922da6`. The upstream LFS OID for
  `train_data_prepared.jsonl` is exactly the SHA-256 of the repository file, so
  no preprocessing step is inferred. The upstream dataset card declares MIT.
- The two RouterBench JSON partitions derive from
  `routerbench_5shot.pkl` at revision
  `784021482c3f320c6619ed4b3bb3b41a21424fcb`. After parsing list literals,
  joining message elements with LF, and mapping five supported route targets,
  30,378 of 30,379 records agree with `oracle_model_to_route_to`. The sole
  exception is `mbpp.dev.101`: WizardLM, Mixtral, and CodeLlama all score 1.0;
  the local file keeps the WizardLM response while the source oracle field
  names Mixtral. All 6,856 non-multiple-choice prompt/response pairs match
  exactly. The official code repository is MIT, but the upstream dataset card
  does not declare a data license, so prepared RouterBench data must not be
  assumed redistributable.
- BookSum is pinned to revision
  `c62321036e5647db5767ecaff139912b554dc938`. All 9,600 rows of `train.csv`
  collapse to 4,845 unique `chapter_path` values; duplicate paths contain
  identical chapter text. Mapping each path to its chapter and applying the
  BooookScore 2048-token `cl100k_base` chunker reproduces all 4,845 local chunk
  lists exactly. The BookSum card declares BSD-3-Clause and separately states
  research-use and underlying-rights terms that still require release review.
- `all_books_chunked_2048.pkl` uses the exact filename and command documented
  by BooookScore revision
  `094bf69bc55317b728b4b2bb679c287e0176ae6c`: `python -m
  booookscore.chunk --chunk_size 2048 --input_path all_books.pkl --output_path
  all_books_chunked_2048.pkl`. The chunker uses `cl100k_base` and the upstream
  requirement pins `tiktoken==0.5.2`. The private `all_books.pkl` was not
  available for byte comparison, and the project states that the dataset is
  released only to academic labs through its request form. The MIT code
  license is not treated as a data redistribution grant.

The local RouterBench creation commits are also recorded in the manifest. The
BookSum comparison establishes semantic equality of the prepared object; it
does not claim that pickle serialization bytes are invariant across Python
versions. Exact prepared-file checksums remain authoritative for timed paper
runs.

Run `bash artifact/scripts/prepare_data.sh` to verify all resolved targets.
For a missing prepared file, pass an explicit
`--source DATASET_ID=/path/to/file` together with `--strategy link` or
`--strategy copy`. The source is checksum-verified before staging, a valid
existing target is reused, and an existing mismatch is never overwritten.
This stages known prepared inputs. It deliberately does not download large or
access-controlled data implicitly, and it does not bypass upstream terms. For
timed runs, a reconstructed object must still match the prepared-file checksum
in `manifests/datasets.yaml` before it is accepted.

A clean checkout can acquire the exact public MixInstruct object without a
manual browser step:

```bash
bash artifact/scripts/prepare_data.sh \
  --dataset mixinstruct_prepared --download-public
```

The operation is opt-in, revision-pinned, and checksum-gated. It defaults to a
link from the ignored repository destination into the Hugging Face cache; pass
`--strategy copy` to materialize a copy. The implementation refuses automatic
download for every manifest entry that lacks `automatic_download: allowed`, so
the flag cannot bypass the unresolved RouterBench license or restricted/private
BookSum and BooookScore workflows.

Run `bash artifact/scripts/prepare_models.sh` to verify every pinned revision
in the Hugging Face cache. Downloads occur only with both `--download` and an
explicit `--acknowledge-access-terms`; gated model terms and authentication
remain the evaluator's responsibility. The acknowledgement cannot override an
unresolved manifest license: Alpaca-13B automatic download is refused pending
manual review. A requested model is always fetched at the exact revision in
`models.yaml`. Verification
also requires a valid JSON object in `config.json`, at least one model weight
file, no broken cache links, matching snapshot/weight layout, and matching
configuration and license-evidence hashes; a directory bearing the correct
revision name alone is not accepted as a complete snapshot.

## Cost-profile policy

The parameterized offline builder is documented in
[resource_generation.md](resource_generation.md). It supports explicit model,
log, execution-plan and output selections, and writes new resources without
appending to legacy source files. Its `plan` action records acquisition commands
only; its `build` action needs no GPU or model loading. Synthetic tests and
reconstruction of the retained ChatGLM/StableLM measurements are supported.

The default legacy entry point selects `--machine_name zxcpu` and consumes the
frozen `get_my_cost_table_directly_zxcpu.py` table. Both serialized table
modules are required because `search_exec_plans.py` imports them eagerly; the
doctor hashes those runtime files and the table loader directly. The frozen
table can be used without the original profiling logs.

The retained command archive recovers the common profiler invocation at
commit `43561cb0af9128d440aecb0f2bfdadfa6c0079a6`: synthetic 16/16 lengths,
one request, eager mode, TP 1/2/4, temperature 1.0, GPU ratio 0.9, weight-load
degree 2, and the vLLM backend. The table builder uses two profiles per point,
a 1 ms threshold, and sample configuration `(1, 1, -1, 0)`. The zxcpu
serialized table was last updated by
`de1677dcb7e2a216172aedf3262b19b106f17786`.

The serialized table identifies the runtime cost data. The complete zxcpu
raw-log set and an immutable GPU topology record are unavailable, so fresh
profile reproduction remains unverified. Raw archives are optional provenance;
a missing archive is a doctor warning,
while a missing or changed serialized runtime table remains an error.

Cost profiling uses ordinary Hugging Face authentication (`HF_TOKEN` or the
local token store). Personal access tokens are not embedded in the profiler.

## CPU scheduling input bundle

`artifact/inputs/frozen/*.json` contains numeric request IDs, token lengths,
application dependencies and source hashes produced by the original workload
builders. The seven bundles contain no request text or model weights.
`artifact/inputs/model_configurations.json` freezes the 14 pinned model
configuration objects needed by the CPU cost simulator. These files permit
offline scheduling from a clean checkout; they do not grant permission to
redistribute upstream texts or weights. The access requirements above still apply.
Experimental logs, result indexes and plotting measurements are not distributed.
