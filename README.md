## Requirements

- Python 3.8 or later
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- SciPy
- scikit-learn
- A C++17 compiler compatible with the installed PyTorch version

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require platform-specific installation commands. If necessary, install them first using the instructions for the target CUDA or CPU environment, and then install the remaining requirements.

## Building the C++ Extension

CongMatch requires the fused association extension. Build it from this directory:

```bash
cd congmatch
python setup.py build_ext --inplace
```

The build creates an `_association_fused_cpp` shared library in the current directory.

The extension is implemented by two translation units:

- `association_fused_candidates_cpp.cpp` performs label-based DualSim filtering, produces candidate query-target node pairs, reports empty candidate sets, and defines the Python bindings.
- `association_fused_build_cpp.cpp` converts the candidate pairs into association-graph edges. Two candidate assignments are connected when they can coexist injectively and preserve the directed query-edge constraints in the target graph.

Both files are compiled into the same `_association_fused_cpp` extension.

## Data

### TU datasets

TU datasets can be selected by short name, for example `cox2`, `aids`, or `mutag`. They are stored or downloaded under the repository-level `dataset` directory.

### GRF datasets

A GRF dataset is selected by passing its root directory. The directory must contain:

```text
dataset_root/
|-- queries/
|   |-- query_0.grf
|   |-- query_0_1.grf
|   `-- ...
`-- targets/
    |-- target_0.grf
    `-- ...
```

`query_<id>.grf` is the positive query for `target_<id>.grf`. Files named `query_<id>_<negative_id>.grf` are negative queries for the same target.

GRF file examples are in the `datasets/synthetic` folder.

## Usage

Run on a TU dataset:

```bash
python run.py --dataset_name cox2
```

Run on a GRF dataset:

```bash
python run.py --dataset_name /path/to/dataset_root
```

An equivalent command that states the fixed configuration explicitly is:

```bash
python run.py \
  --dataset_name cox2 \
  --candidate_pi_mode association_fused \
  --association_backend transformer_dual_relation \
  --batch_size 16 \
  --label_pi_mode off \
  --candidate_loss bce \
  --camh_enable true \
  --camh_loss_type camh_new
```

## Output

Each run creates a separate directory under `experiments` by default. The directory contains:

- `config.json`: resolved command-line configuration
- `best.pt`: model parameters from the best validation epoch
- `result.json`: training, validation, test, timing, and association-graph statistics

Use `--output_dir` to change the output location. Use `--save_model false` to disable `best.pt` creation.

## Project Structure

```text
congmatch/
|-- association_fused_build_cpp.cpp
|-- association_fused_candidates_cpp.cpp
|-- candidate_graph.py
|-- dataset.py
|-- inference.py
|-- model.py
|-- requirements.txt
|-- run.py
`-- setup.py
```

`candidate_graph.py` loads the compiled extension, batches association graphs, implements the dual-relation Transformer, and computes CAMH-new constraint terms. `run.py` provides the training entry point.
