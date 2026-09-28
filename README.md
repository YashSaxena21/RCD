# Retriever Adapters as Teachers: Representation Correction Distillation for Dense Retrieval

This repository contains the implementation of **Representation Correction
Distillation (RCD)** described in the paper draft *Retriever Adapters as Teachers:
Representation Correction Distillation for Dense Retrieval*.

RCD uses a lightweight retriever adapter trained with limited target-domain
relevance supervision as a teacher for a standalone dense retriever. The teacher is
frozen before distillation. The student then learns from the teacher over both labeled
and additional unlabeled target queries, without using relevance labels in the student
loss.

The key distinction from conventional representation distillation is the target being
matched. RCD transfers the **adapter-induced correction** to the base retrieval space,
rather than requiring the student to reproduce the teacher's complete final
representation.

At inference time, retrieval uses only the fine-tuned dense retriever. The adapter
teacher and training-only projection are discarded.

## Method

Let `f_0` be a frozen pretrained retriever, `g_psi` a retriever adapter, and `f_theta`
the student initialized from `f_0`. The target-domain training pool is divided into a
small labeled set `Q_L` and an additional unlabeled set `Q_U`.

1. Train `g_psi` on relevance labels from `Q_L` while keeping `f_0` frozen.
2. Freeze the trained adapter teacher.
3. Retrieve a fixed candidate set for each query with `f_0`.
4. Train `f_theta` on query texts from both `Q_L` and `Q_U` using only frozen teacher
   representations and scores.
5. Evaluate `f_theta` as a standard dense bi-encoder.

For query `q` and candidate document `d`, the frozen base retriever produces
representations `q_0` and `d_0`. The adapter teacher produces `q_T` and `d_T`, and the
student produces `q_theta` and `d_theta`. Their representation corrections are:

```text
teacher:  delta_q_T     = q_T - q_0       delta_d_T     = d_T - d_0
student:  delta_q_theta = q_theta - q_0   delta_d_theta = d_theta - d_0
```

### Representation correction

A shared linear projection maps student corrections into the teacher adapter space.
RCD aligns correction directions using cosine distance:

```text
L_query = 1 - cos(P(delta_q_theta), delta_q_T)

L_document = mean_d [1 - cos(P(delta_d_theta), delta_d_T)]

L_correction = beta_q * L_query + beta_d * L_document
```

Documents are averaged within each query before the batch reduction, so each query
receives equal weight. The projection is optimized jointly with the student and is not
used during retrieval.

### Ranking distillation

Teacher and student scores are converted into distributions over the same fixed
candidate set using temperature `tau`. Ranking behavior is transferred with listwise
KL divergence:

```text
L_ranking = KL(p_teacher(. | q) || p_student(. | q))
```

The full implementation objective is:

```text
L_RCD = L_ranking + alpha * L_correction
```

The weights are configurable with `--alpha`, `--beta_q`, and `--beta_d`.

### RCD variants

| Paper name | CLI key | Ranking distillation | Correction alignment |
| --- | --- | ---: | ---: |
| Ranking Only | `RANKING_ONLY` | yes | no |
| Correction Only | `CORRECTION_ONLY` | no | yes |
| Full RCD | `FULL_RCD` | yes | yes |

## Supervision Protocol

RCD is **semi-supervised at the complete pipeline level** because relevance labels are
used to train the adapter teacher. Student post-training itself is label-free.

During student training:

- candidate documents are retrieved by the frozen base retriever;
- relevant documents are not injected into candidate sets;
- qrel scores and relevant-document identities do not enter the student objective;
- the teacher's top-ranked documents are determined only by teacher scores;
- qrels are used only to recover benchmark splits, select checkpoints on validation
  data, and evaluate final retrieval performance.

The default student setting uses all adaptation-pool query texts from `Q_L` and `Q_U`.
Validation and test queries are excluded from both teacher and student training.

Runs record ordered candidate-index and candidate-ID fingerprints together with a
corpus-order fingerprint. These records allow RCD, Margin-MSE, and EmbedDistill runs to
be checked for exact candidate-set equality.

## Paper Setup

The primary paper configuration uses:

- `intfloat/e5-large-v2` as the base retriever and student initialization;
- IMRNNs as the adapter teacher;
- teacher relevance-supervision budgets of 10%, 30%, and 50%;
- three optimization seeds: 42, 43, and 44;
- frozen base-retriever top-64 candidate pools;
- at most 32 training candidates per query;
- nDCG@8 as the primary metric;
- retrieval cutoffs `1 2 4 8 16 32 64` in the released implementation.

The paper also studies transfer across adapter architectures with Search-Adaptor and
across retriever backbones with `Qwen/Qwen3-Embedding-0.6B`.

### Benchmarks

| Dataset | Domain / task | Retrieval type | Documents | Adaptation queries | 10% | 30% | 50% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ContractNLI | Legal contracts | Multi-hop | 95 | 783 | 78 | 235 | 392 |
| PrivacyQA | Privacy | Multi-hop | 7 | 156 | 16 | 47 | 78 |
| ComLQ | Complex logical queries | Mixed | 11,251 | 2,329 | 233 | 699 | 1,165 |
| SciFact | Scientific evidence | Single-hop | 5,183 | 728 | 73 | 219 | 364 |
| NFCorpus | Biomedical | Single-hop | 3,633 | 2,590 | 259 | 777 | 1,295 |

The adaptation pool is 80% of each benchmark's queries; validation and test each use
10%. Teacher subsets are deterministic and nested: the 10% subset is contained in the
30% subset, which is contained in the 50% subset.

### Paper result snapshot

The current draft reports the following macro-average relative changes in nDCG@8 over
the pretrained E5 retriever across the five benchmarks:

| Method | 10% supervision | 30% supervision | 50% supervision |
| --- | ---: | ---: | ---: |
| IMRNN teacher | +4.3% | +10.6% | +17.3% |
| Supervised E5 | +0.8% | +0.8% | +7.1% |
| Margin-MSE | -13.1% | -2.1% | +1.2% |
| EmbedDistill | -36.1% | -37.2% | -22.8% |
| **Full RCD** | **+6.9%** | **+13.3%** | **+21.7%** |

At 50% supervision, the draft reports a macro-average nDCG@8 of 0.504 for Full RCD,
3.7% higher than the frozen IMRNN teacher. Refer to the paper for dataset-level
results, ComLQ logical-structure and hop-count slices, query-scope controls, and
teacher/backbone generalization experiments.

## Included Methods

The repository exposes the following commands:

| Command | Method |
| --- | --- |
| `rcd-train rcd` | Ranking Only, Correction Only, or Full RCD |
| `rcd-train margin-mse` | Teacher-top versus remaining-candidate score-margin matching |
| `rcd-train embeddistill` | Final teacher representation matching plus score distillation |
| `rcd-train supervised` | Supervised fine-tuning of the same dense retriever |
| `rcd-train base` | Evaluation of the pretrained retriever |

The paper evaluates additional domain-adaptation baselines outside this focused
release. They are not reimplemented here.

Supported retriever backbones:

- `intfloat/e5-large-v2`
- `Qwen/Qwen3-Embedding-0.6B`

Supported adapter teachers:

- IMRNNs
- Search-Adaptor

Margin-MSE, EmbedDistill, and RCD use the same frozen teacher, candidate sets, student
initialization, and optimization budget when run with the same configuration.

## Installation

Python 3.10 through 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[faiss]'
```

Install a CUDA-enabled PyTorch wheel appropriate for the target cluster if the default
resolver does not select one. The IMRNNs integration uses `imrnns==0.2.3`.

Verify the command-line entry points:

```bash
rcd-train --help
rcd-train rcd --help
rcd-prepare-beir --help
```

## Data

The standard input layout is:

```text
dataset/
|-- corpus.jsonl
|-- queries.jsonl
`-- qrels/
    |-- train.tsv
    |-- val.tsv
    `-- test.tsv
```

Corpus records use `_id`, optional `title`, and `text`. Query records use `_id`,
`text`, and `type`; datasets without typed queries can use `"type": "retrieval"`.
Qrels may use the standard BEIR three-column TSV format with a header.

The pipeline includes loaders for ComLQ and LegalBench-RAG ContractNLI/PrivacyQA.
SciFact and NFCorpus can be converted into deterministic shared low-resource splits:

```bash
rcd-prepare-beir \
  --datasets scifact nfcorpus \
  --data_root datasets \
  --output_root prepared/beir \
  --split_seed 42 \
  --fraction_seed 42 \
  --fractions 0.10 0.30 0.50
```

Use the same prepared split directory and fraction manifest for every compared method.
For a same-budget supervised run, its training qrels must contain exactly the selected
`Q_L` subset rather than the complete adaptation pool.

## Running Full RCD

The following command runs the primary E5/IMRNN configuration on ComLQ at 50%
teacher supervision:

```bash
rcd-train rcd \
  --encoder e5 \
  --teacher imrnns \
  --teacher-fraction 0.50 \
  --teacher-fraction-seed 42 \
  --dataset_format comlq \
  --dataset_dir datasets/comlq \
  --variants FULL_RCD \
  --seeds 42 43 44 \
  --candidate_pool_k 64 \
  --max_train_candidates_per_query 32 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/e5-imrnns-comlq-f050
```

If the dataset- and seed-specific IMRNN checkpoint is absent, the RCD command trains
it before student distillation. Pass `--no_auto_train_teacher_if_missing` to require a
pre-existing checkpoint. For multi-dataset runs, use
`--teacher_checkpoint_template` with `{dataset}` and `{seed}` placeholders; never
reuse one dataset's teacher for another dataset.

Run all three public variants by changing the variant argument:

```bash
--variants RANKING_ONLY CORRECTION_ONLY FULL_RCD
```

Examples for E5/IMRNNs, Qwen/Search-Adaptor, and Slurm are available under
[`examples/`](examples/).

## Baselines

Use the same teacher checkpoint for all teacher-distillation methods in a controlled
comparison:

```bash
rcd-train margin-mse \
  --encoder e5 --teacher imrnns --teacher-fraction 0.50 \
  --dataset_format comlq --dataset_dir datasets/comlq \
  --teacher_checkpoint_path runs/teachers/imrnns-e5-comlq-f050-seed42.pt \
  --seeds 42 --ks 1 2 4 8 16 32 64 \
  --output_dir runs/margin-mse-comlq-f050

rcd-train embeddistill \
  --encoder e5 --teacher imrnns --teacher-fraction 0.50 \
  --dataset_format comlq --dataset_dir datasets/comlq \
  --teacher_checkpoint_path runs/teachers/imrnns-e5-comlq-f050-seed42.pt \
  --seeds 42 --ks 1 2 4 8 16 32 64 \
  --output_dir runs/embeddistill-comlq-f050
```

Margin-MSE and EmbedDistill fail when their requested teacher is missing unless
`--auto_train_teacher_if_missing` is explicitly supplied to those commands.

Supervised and base-retriever evaluation use the same encoder interface:

```bash
rcd-train supervised \
  --encoder e5 \
  --dataset_format standard \
  --dataset_dir datasets/comlq-f050-supervised \
  --dataset_name comlq \
  --seed 42 --device cuda --fp16 \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/supervised-e5-comlq-f050

rcd-train base \
  --encoder e5 \
  --dataset_format standard \
  --dataset_dir datasets/comlq \
  --dataset_name comlq \
  --seed 42 --device cuda \
  --ks 1 2 4 8 16 32 64 \
  --output_dir runs/e5-base-comlq
```

## Reproducibility and Outputs

Each seed directory stores:

- latest and best training checkpoints;
- the standalone student model;
- `final_results.json` and `final_results.csv`;
- training logs and run metadata;
- teacher and candidate fingerprints;
- the complete experiment configuration.

Multi-seed runs also write per-seed and aggregate mean/standard-deviation files.
ComLQ runs include logical-query and hop-count test breakdowns.

Resume is enabled by default. The run contract includes the scientific configuration,
teacher checkpoint hash, selected teacher-query subset, and candidate fingerprints. A
contract mismatch fails explicitly instead of silently loading incompatible state. Use
a new output directory with `--no_resume` when intentionally starting a different
experiment.

## Cluster Use

A Slurm template is provided at
[`examples/slurm/train_rcd.slurm`](examples/slurm/train_rcd.slurm). The default student
configuration uses microbatching, gradient checkpointing, mixed precision, and bounded
candidate sets. On a 44 GB L40S, the paper configuration starts with:

```text
--batch_size 1
--micro_batch_size 1
--max_train_candidates_per_query 32
```

## Tests

```bash
python -m pytest
python -m rcd_retrieval.margin_mse --run_loss_tests
python -m rcd_retrieval.embeddistill --run_loss_tests
rcd-train rcd --teacher search-adaptor --run_search_adaptor_tests
rcd-prepare-beir --run_tests
```

The tests cover objective invariants, masking, label rejection in student losses,
variant contracts, strict resume behavior, candidate fingerprints, deterministic
teacher fractions, exact teacher-subset validation, and encoder configuration. A real
dry run requires benchmark data and a matching teacher checkpoint.

## References

- Yoon et al. [Search-Adaptor: Embedding Customization for Information
  Retrieval](https://aclanthology.org/2024.acl-long.661/), ACL 2024.
- Kim et al. [EmbedDistill: A Geometric Knowledge Distillation for Information
  Retrieval](https://arxiv.org/abs/2301.12005), 2023.
- Zhang et al. [Qwen3 Embedding: Advancing Text Embedding and Reranking through
  Foundation Models](https://arxiv.org/abs/2506.05176), 2025.
- [IMRNNs Python package](https://pypi.org/project/imrnns/).

Citation information for RCD will be added when the paper is released.
