"""
CLI entry: MERFISH DAX segmentation, training data generation, Cellpose training, scoring.
This code was originally built by hand. Used Cursor to help with refactoring after phase 1.
See cursor_code_readability_in_main.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from cellpose import io, models, train
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cellpose_infer import build_cellpose_model, predict_masks_from_epi_stack
from dax_preprocessing import (
    generate_mix,
    generate_z_stack,
    load_dax,
    parse_float_list,
    slices_for_z_plane,
)
from generate_train_submission_v2 import build_submission
from my_paths import CUSTOM_DATA, MODELS_DIR, RESULTS, TRAIN, provided_code
from provided_code.metric import score

CLI_MODES = (
    "kaggle-infer",
    "test-infer",
    "test-score",
    "gen-data",
    "train",
)

EXAMPLES = """\
examples:
  python main.py test-infer --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0160
  python main.py test-score --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0160
  python main.py kaggle-infer --test-mode average-z_dapi-polyt --model_name my_new_model_epoch_0040
  python main.py gen-data --custom_data_dir dapi-polyt
  python main.py train --custom_data_dir dapi-polyt --epochs 100 --batch_size 10
"""


@dataclass
class PipelineConfig:
    seed: int = 42
    diam: float = 20.0
    z_single: int = 2
    z_planes: list[int] = field(default_factory=lambda: [2])


def parse_args():
    parser = argparse.ArgumentParser(
        description="MERFISH / Cellpose pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "mode",
        choices=CLI_MODES,
        help="Pipeline step to run",
    )
    parser.add_argument("--epochs", default=100, help="training epochs")
    parser.add_argument("--test-mode", help="preprocessing preset id (e.g. dapi-polyt)")
    parser.add_argument(
        "--batch_size",
        default=10,
        help="training batch size (e.g. 8 uses ~7GB GPU for Cellpose)",
    )
    parser.add_argument("--custom_data_dir", default="", help="subfolder under custom_dataset/")
    parser.add_argument(
        "--model_name",
        default=None,
        help="checkpoint folder name under models/<test-mode>/models/",
    )
    return parser.parse_args()


def list_train_fovs(train_root: Path) -> list[str]:
    return sorted(fov for fov in os.listdir(train_root) if "FOV_" in fov)


def split_fovs(train_root: Path, seed: int) -> tuple[list[str], list[str], list[str]]:
    all_fovs = list_train_fovs(train_root)
    train_fovs, test_fovs = train_test_split(all_fovs, train_size=0.9, random_state=seed)
    val_fovs = test_fovs
    return list(train_fovs), list(test_fovs), list(val_fovs)


def kaggle_masks_dir(results: Path, format_id: str, model_name: str | None) -> Path:
    return results / "kaggle" / format_id / (model_name if model_name else "default")


def aggregate_cell_boundary(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values(["image_row", "image_col"])
    return pd.Series(
        {
            "boundaryX_z2": group["boundaryX_z2"].iloc[0],
            "boundaryY_z2": group["boundaryY_z2"].iloc[0],
        }
    )


def run_kaggle_infer(args, cfg: PipelineConfig) -> None:
    format_id = args.test_mode
    model = build_cellpose_model(
        preset="kaggle",
        format_id=format_id,
        model_name=args.model_name,
        diam_mean=cfg.diam,
    )
    kaggle_fovs = list_train_fovs(provided_code / "test")

    out_root = kaggle_masks_dir(RESULTS, format_id, args.model_name)
    out_root.mkdir(parents=True, exist_ok=True)

    for fov in tqdm(kaggle_fovs, desc="Kaggle inference"):
        fov_num = fov.split("_")[1]
        stack_path = (
            provided_code / "test" / fov / f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_num}.dax"
        )
        epi_stack = load_dax(stack_path)
        print(f"Epi stack shape: {epi_stack.shape}  (frames, height, width)")
        masks = predict_masks_from_epi_stack(epi_stack, model, format_id, cfg.z_single)
        tqdm.write(f"FOV: {fov},  Number of cells found: {masks.max()}")
        np.save(out_root / f"{fov}_mask.npy", masks)


def run_test_infer(args, cfg: PipelineConfig, test_fovs: list[str]) -> None:
    format_id = args.test_mode
    model = build_cellpose_model(
        preset="local_test",
        format_id=format_id,
        model_name=args.model_name,
        diam_mean=cfg.diam,
    )

    result_dir = RESULTS / format_id
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    for fov in tqdm(test_fovs, desc="Test inference"):
        fov_num = fov.split("_")[1]
        epi_stack = load_dax(
            TRAIN / fov / f"Epi-750s5-635s5-545s1-473s5-408s5_{fov_num}.dax"
        )
        for z_plane in cfg.z_planes:
            masks = predict_masks_from_epi_stack(epi_stack, model, format_id, z_plane)
            tqdm.write(f"FOV: {fov},  Number of cells found: {masks.max()}")
            np.save(result_dir / f"{fov}_z{z_plane}_mask.npy", masks)


def run_test_score(args, cfg: PipelineConfig, test_fovs: list[str]) -> None:
    format_id = args.test_mode
    print("Loading spots_train_w_cell_id_solution.csv ...")
    train_solution_df = pd.read_csv("results/spots_train_w_cell_id_solution.csv")
    train_solution_df = train_solution_df[train_solution_df["fov"].isin(test_fovs)]
    print(f" {len(train_solution_df):,} spots across {train_solution_df['fov'].nunique()} FOVs")

    average_score_across_z = 0.0
    for z_level in cfg.z_planes:
        sub_train_solution_df = train_solution_df[train_solution_df["global_z"] == float(z_level)]
        if args.verbose:
            print(f"\nLoading masks at z level {z_level} ...")
        masks = {
            fov: np.load(RESULTS / format_id / f"{fov}_z{z_level}_mask.npy") for fov in test_fovs
        }
        submit_df = build_submission(masks, sub_train_solution_df)
        score_at_z = score(sub_train_solution_df, submit_df, "spot_id")
        average_score_across_z += score_at_z
        if args.verbose:
            print(f"  Score on z level {z_level}: {score_at_z}")

    print("==== Final Results ====")
    print(f"Format: {format_id}")
    print(f"Final Score average_score_across_z: {average_score_across_z / len(cfg.z_planes)}\n")


def run_gen_data(args, train_fovs: list[str], val_fovs: list[str]) -> None:
    custom_data_dir = CUSTOM_DATA / args.custom_data_dir
    for sub in ("train", "val"):
        p = custom_data_dir / sub
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    train_solution_df = pd.read_csv("results/spots_train_w_cell_id_solution.csv")
    cell_boundaries_train_df = pd.read_csv(
        provided_code / "train/ground_truth/cell_boundaries_train.csv"
    )
    cell_boundaries_train_df.iloc[:, 1:] = cell_boundaries_train_df.iloc[:, 1:].applymap(
        parse_float_list
    )
    cell_boundaries_train_df.rename({"Unnamed: 0": "gt_cluster_id"}, inplace=True, axis=1)
    reference_xy = pd.read_csv(provided_code / "reference/fov_metadata.csv")

    for fov_list, split_name in ((train_fovs, "train"), (val_fovs, "val")):
        for fov in tqdm(fov_list, desc=f"gen-data {split_name}"):
            fov_num = fov.split("_")[1]
            train_solution_df_fov = train_solution_df[
                (train_solution_df["fov"] == fov) & (train_solution_df["global_z"] == 2.0)
            ]
            reference_xy_fov = reference_xy[reference_xy["fov"] == fov]
            solution_cells_df = cell_boundaries_train_df.merge(
                train_solution_df_fov, how="inner", on="gt_cluster_id"
            )
            solution_cells_df = solution_cells_df[
                [
                    "gt_cluster_id",
                    "boundaryX_z2",
                    "boundaryY_z2",
                    "image_row",
                    "image_col",
                    "fov",
                    "spot_id",
                ]
            ]

            apply_solution_cells_df = (
                solution_cells_df.groupby("gt_cluster_id", group_keys=False)
                .apply(aggregate_cell_boundary)
                .reset_index()
            )

            master_mask = np.zeros((2048, 2048), dtype=np.int32)

            for i in range(len(apply_solution_cells_df)):
                x_data = apply_solution_cells_df.loc[i, "boundaryX_z2"]
                y_data = apply_solution_cells_df.loc[i, "boundaryY_z2"]
                x_px = (
                    2048 - (np.array(x_data) - np.array(reference_xy_fov["fov_x"]))
                    / np.array(reference_xy_fov["pixel_size"])
                )
                y_px = (
                    (np.array(y_data) - np.array(reference_xy_fov["fov_y"]))
                    / np.array(reference_xy_fov["pixel_size"])
                )
                polygon_points = np.array(list(zip(y_px, x_px)), dtype=np.int32)
                cv2.fillPoly(master_mask, [polygon_points], color=int(i + 1))

            cv2.imwrite(
                str(custom_data_dir / split_name / f"cells_{fov_num}_masks.png"),
                master_mask,
            )

            epi_stack = load_dax(
                TRAIN / fov / f"Epi-750s5-635s5-545s1-473s5-408s5_{fov.split('_')[1]}.dax"
            )
            zp = 2
            dapi, polyt = slices_for_z_plane(epi_stack, zp)

            if args.custom_data_dir == "dapi-polyt":
                save_image = generate_mix(dapi, polyt) * 255
            elif args.custom_data_dir == "average-z_dapi-polyt":
                save_image = (
                    generate_mix(
                        generate_z_stack(6, epi_stack),
                        generate_z_stack(5, epi_stack),
                    )
                    * 255
                )
            else:
                raise ValueError(
                    f"custom_data_dir must be 'dapi-polyt' or 'average-z_dapi-polyt', "
                    f"got {args.custom_data_dir!r}"
                )

            save_image = save_image[..., np.newaxis]
            save_image = np.tile(save_image, 3)
            cv2.imwrite(
                str(custom_data_dir / split_name / f"cells_{fov_num}_img.png"),
                save_image,
            )


def run_train(args, cfg: PipelineConfig) -> None:
    custom_data_dir = CUSTOM_DATA / args.custom_data_dir
    model_dir = MODELS_DIR / args.custom_data_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dir = (Path.cwd() / custom_data_dir / "train").as_posix() + "/"
    val_dir = (Path.cwd() / custom_data_dir / "val").as_posix() + "/"
    print(train_dir, val_dir)

    io.logger_setup()
    images, labels, image_names, test_images, test_labels, image_names_test = (
        io.load_train_test_data(
            train_dir,
            val_dir,
            image_filter="_img",
            mask_filter="_masks",
            look_one_level_down=False,
        )
    )

    model = models.CellposeModel(gpu=True, diam_mean=cfg.diam)
    train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=0.1,
        learning_rate=1e-5,
        save_every=20,
        save_each=True,
        batch_size=int(args.batch_size),
        n_epochs=int(args.epochs),
        model_name="my_new_model",
        save_path=model_dir,
        channels=[0, 0],
    )


def main():
    args = parse_args()
    cfg = PipelineConfig()
    train_fovs, test_fovs, val_fovs = split_fovs(TRAIN, cfg.seed)

    match args.mode:
        case "kaggle-infer":
            run_kaggle_infer(args, cfg)
        case "test-infer":
            run_test_infer(args, cfg, test_fovs)
        case "test-score":
            run_test_score(args, cfg, test_fovs)
        case "gen-data":
            run_gen_data(args, train_fovs, val_fovs)
        case "train":
            run_train(args, cfg)


if __name__ == "__main__":
    main()
