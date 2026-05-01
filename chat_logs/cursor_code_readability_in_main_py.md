# Code readability in main.py
_Exported on 5/1/2026 at 17:21:31 EDT from Cursor (3.2.11)_

---

**User**

refactor the @main.py such it is easier to read and use.

---

**Cursor**

Exploring `main.py` and its dependencies to map structure and propose a readable refactor plan.



Checking dataset.py for overlap and verifying the train path concatenation issue:



Identifying structural issues: module-level execution (`args` + FOV splitting before dispatch), duplicated inference between `kaggle-infer` and `test-infer`, a probable bug in the `average-z_dapi-polyt` branch of `test-infer`, and unclear `train` paths (`os.getcwd() / Path`). Creating a refactor plan.



---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

Implementing the refactor: starting with extracting modules and reading `main.py`.



Creating `dax_preprocessing.py` and `cellpose_infer.py`, then rewriting `main.py`.



Writing the refactored `main.py`.



Fixing compatibility: avoiding `include_groups=False` for older pandas, removing unused imports, and verifying `groupby.apply` behavior.



Fixing match/case: bare names in `case` are capture patterns, not constants.



Preserving inference parity: original code used Cellpose eval `diameter` default of 30 while `diam_mean` was 20. Removing explicit `diameter=cfg.diam` from inference calls.



Here’s what was implemented:

### New modules

- **[`dax_preprocessing.py`](dax_preprocessing.py)** — `load_dax`, normalization/mix/stack helpers, `parse_float_list`, `slices_for_z_plane`, plus frame helpers (`poly_t_frame`, `dapi_frame`, constants).
- **[`cellpose_infer.py`](cellpose_infer.py)** — `build_cellpose_model(..., preset="kaggle"|"local_test")`, `cellpose_model_test_format`, `predict_masks_from_epi_stack` (single place for **`average-z_dapi-polyt` vs single-z logic**).

### [`main.py`](main.py)

- **`main()` + `if __name__ == "__main__"`** — no argparse or FOV split on import (`import main` only loads modules).
- **`PipelineConfig`** — `seed`, `diam`, `z_planes`, etc.
- **Explicit imports** from `my_paths` (no `*`).
- **Dispatch** — `match args.mode` → `run_kaggle_infer` / `run_test_infer` / `run_test_score` / `run_gen_data` / `run_train`.
- **`aggregate_cell_boundary`** — moved out of `gen-data` nested def.
- **Argparse** — `choices=CLI_MODES` and **`EXAMPLES`** epilog (`--help`).
- **`kaggle_masks_dir`** — **`model_name=None` → `.../default/`** instead of crashing on `/None`.

### Fixes (from plan)

1. **`test-infer`** — `average-z_dapi-polyt` uses the same averaging path as kaggle via `predict_masks_from_epi_stack`.
2. **Train dirs** — `(Path.cwd() / custom_data_dir / "train").as_posix() + "/"` (and same for `val`).
3. **Inference `diameter`** — still defaults to **30** in `predict_masks_from_epi_stack` to match previous `model.eval(...)` usage; **`diam_mean`** for the model stays **`cfg.diam` (20)**.

Verified in the project venv: `import main` succeeds and `python main.py --help` shows modes and examples.

