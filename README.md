# Representation Correction Distillation

This repository trains a plain dense retriever from a frozen adapter teacher. It contains the RCD objective, IMRNNs and Search-Adaptor teacher backends, Margin-MSE and EmbedDistill baselines, supervised fine-tuning, and pretrained-retriever evaluation.

Supported student encoders:

- `intfloat/e5-large-v2`
- `Qwen/Qwen3-Embedding-0.6B`

The projection used by RCD and EmbedDistill is training-only. Every reported student is evaluated as an ordinary bi-encoder with normalized native embeddings; no teacher or projection head is needed at retrieval time.

## Methods

RCD uses the teacher scores for listwise ranking distillation and aligns changes in the student's representation with changes induced by the teacher:

```text
student delta:  f_theta(x) - f_0(x)
teacher delta:  z_T(x) - z_0(x)
```

The public RCD variants are:

| Variant | CLI key | Ranking KD | Correction alignment |
| --- | ---: | ---: |
| Ranking Only | `RANKING_ONLY` | yes | no |
| Correction Only | `CORRECTION_ONLY` | no | yes |
| Full RCD | `FULL_RCD` | yes | yes |

The baselines hold the encoder, candidate sets, optimization budget, and teacher fixed:

- **Margin-MSE** matches teacher score margins. Its default pair construction is teacher-top versus every other valid candidate.
- **EmbedDistill** matches the teacher's final query/document representations through one shared linear projection and includes listwise score distillation by default.
- **Supervised** fine-tunes the same retriever using qrel positives.
- **Base** evaluates the pretrained E5 or Qwen retriever without fine-tuning.

### Supervision boundary

IMRNNs and Search-Adaptor teachers are trained with qrels. `--teacher-fraction` controls the nested subset used to fit the teacher. Student post-training remains unlabeled:

- candidates come from the frozen base retriever;
- relevant documents are not injected into student candidates;
- qrel scores and identities do not enter RCD, Margin-MSE, or EmbedDistill losses;
- qrels are used for split membership, validation selection, and final evaluation.

The default candidate source is frozen retrieval and dynamic refresh is disabled. Runs save ordered candidate-index and candidate-ID fingerprints plus a corpus-order fingerprint, so compared objectives can be checked for exact candidate equality.

## Installation

Python 3.10-3.12 is supported. Create a clean environment on the GPU node and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[faiss]'
```

Install a CUDA-enabled PyTorch wheel appropriate for the cluster if the default resolver does not select one. The IMRNNs integration is pinned to `imrnns==0.2.3`; the code uses its current `ModelConfig(input_dim=...)`, training, checkpoint, and `forward_with_details` interfaces.

Check the command surface:

```bash
rcd-train --help
rcd-train rcd --help
```

## Data layout

The standard format is:

```text
dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    ├── val.tsv
    └── test.tsv
```

Corpus rows use `_id`, `title` (optional), and `text`. Query rows use `_id`, `text`, and `type`; use `"type": "retrieval"` for datasets without typed queries. Qrels may use the BEIR three-column format with a header.

The core pipeline supports ComLQ and LegalBench-RAG data preparation. `rcd_retrieval.beir_splits` prepares deterministic low-resource splits for SciFact and NFCorpus while preserving their official test sets.

```bash
rcd-prepare-beir \
  --datasets scifact nfcorpus \
  --data_root datasets \
  --output_root prepared/beir
```

## Running RCD

E5 with an IMRNNs teacher at 50% teacher supervision:

```bash
rcd-train rcd \
  --encoder e5 \
  --teacher imrnns \
  --teacher-fraction 0.50 \
  --dataset_format comlq \
  --dataset_dir datasets/comlq \
  --teacher_checkpoint_template 'runs/teachers/comlq/imrnns-e5-f050-seed{seed}.pt' \
  --variants RANKING_ONLY CORRECTION_ONLY FULL_RCD \
  --seeds 42 43 44 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/e5-imrnns-comlq-f050
```

A missing IMRNNs checkpoint is trained by the RCD command. Add `--no_auto_train_teacher_if_missing` to require an existing checkpoint. Margin-MSE and EmbedDistill require explicit `--auto_train_teacher_if_missing` if their checkpoint does not exist.

Qwen with Search-Adaptor:

```bash
rcd-train rcd \
  --encoder qwen \
  --teacher search-adaptor \
  --teacher-fraction 0.50 \
  --dataset_format comlq \
  --dataset_dir datasets/comlq \
  --search_adaptor_datasets comlq \
  --variants FULL_RCD \
  --seeds 42 \
  --output_dir runs/qwen-search-adaptor-comlq-f050
```

Qwen uses the retrieval instruction recommended for Qwen3-Embedding, left padding, 512-token inputs, FP32 master parameters, and AMP compute. The FP32 master parameters avoid the BF16 `GradScaler.unscale_` failure seen with mixed parameter dtypes.

## Baselines

```bash
rcd-train margin-mse \
  --encoder e5 --teacher imrnns --teacher-fraction 0.50 \
  --dataset_format comlq --dataset_dir datasets/comlq \
  --teacher_checkpoint_path runs/teachers/comlq/imrnns-e5-f050-seed42.pt \
  --seeds 42 --output_dir runs/margin-mse

rcd-train embeddistill \
  --encoder e5 --teacher imrnns --teacher-fraction 0.50 \
  --dataset_format comlq --dataset_dir datasets/comlq \
  --teacher_checkpoint_path runs/teachers/comlq/imrnns-e5-f050-seed42.pt \
  --seeds 42 --output_dir runs/embeddistill

rcd-train supervised \
  --encoder e5 --dataset_format standard \
  --dataset_dir datasets/comlq --dataset_name comlq \
  --seed 42 --device cuda --fp16 \
  --output_dir runs/supervised-e5

rcd-train base \
  --encoder qwen --dataset_format standard \
  --dataset_dir datasets/comlq --dataset_name comlq \
  --seed 42 --device cuda \
  --output_dir runs/qwen-base
```

For Search-Adaptor versions of Margin-MSE and EmbedDistill, change `--teacher imrnns` to `--teacher search-adaptor` and provide `--search_adaptor_datasets`.

## Outputs and resume

Each seed writes its own directory with the latest and best training state, model files, `final_results.json`, and `final_results.csv`. Multi-seed runs additionally write per-seed and aggregate mean/std files. Existing latest checkpoints resume by default; use `--no_resume` only when intentionally starting over in a clean output directory.

Do not reuse one teacher path across datasets or seeds. Use `--teacher_checkpoint_template` with `{dataset}` and `{seed}` for suites. Teacher metadata is checked against the dataset, encoder, seed, package version, fraction, fraction-selection seed, and exact selected-query fingerprint before loading.

Resume is strict: every student checkpoint stores a run contract covering the scientific configuration, teacher file hash, and candidate fingerprints. A changed contract fails instead of silently loading stale state; use a clean output directory with `--no_resume` for a new experiment.

## Cluster use

A Slurm template is in [`examples/slurm/train_rcd.slurm`](examples/slurm/train_rcd.slurm). The defaults use microbatching, gradient checkpointing, AMP, and bounded candidate sets. For a 44 GB L40S, the standard starting point is `--batch_size 1 --micro_batch_size 1 --max_train_candidates_per_query 32`.

## Tests

```bash
python -m pytest
python -m rcd_retrieval.margin_mse --run_loss_tests
python -m rcd_retrieval.embeddistill --run_loss_tests
rcd-train rcd --teacher search-adaptor --run_search_adaptor_tests
```

The unit tests cover objective invariants, masking, qrel-label rejection, variant contracts, strict resume behavior, candidate fingerprints, deterministic teacher fractions, exact teacher-subset validation, and encoder configuration. A real dry run still requires a dataset and a matching teacher checkpoint.

## Method references

- Search-Adaptor: [Search-Adaptor: Embedding Customization for Information Retrieval](https://aclanthology.org/2024.acl-long.661/)
- EmbedDistill: [EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval](https://arxiv.org/abs/2301.12005)

The Search-Adaptor teacher in this repository follows the paper's pairwise ranking, reconstruction, and query-prediction objective. It is an adaptation to the shared candidate-conditioned RCD interface; it is not based on the unrelated Elasticsearch wrapper with a similar repository name.
