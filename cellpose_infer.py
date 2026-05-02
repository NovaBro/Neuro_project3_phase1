"""Cellpose inference helpers (format presets and mask prediction)."""

from pathlib import Path

import numpy as np
from cellpose.models import CellposeModel

from dax_preprocessing import generate_mix, generate_z_stack, normalize, slices_for_z_plane

AVERAGE_Z_DAPI_POLYT = "average-z_dapi-polyt"


def pretrained_path_for_format(format_id: str, model_name: str) -> str:
    return str(Path.cwd() / "models" / format_id / "models" / model_name)


def build_cellpose_model(
    *,
    model_type: str,
    format_id: str,
    model_name: str | None,
    diam_mean: float,
) -> CellposeModel:

    if model_name:
        path = pretrained_path_for_format(format_id, model_name)
        print(f"\nLoading Model From: {path}")
        if not Path(path).exists(): exit("MODEL PATH DOESNT EXIST or MODEL DOESNT EXIST: FAIL")
        return CellposeModel(
            model_type=model_type,
            gpu=True,
            pretrained_model=path,
            diam_mean=diam_mean,
        )
    return CellposeModel(model_type=model_type, gpu=True, diam_mean=diam_mean)


def cellpose_model_test_format(
    input_data: list,
    model: CellposeModel,
    format_id: str,
    diameter:int=30,
    channels=(0, 0),
):
    ch = list(channels)
    match format_id:
        case "dapi-data":
            dapi, polyt = input_data
            return model.eval(dapi, diameter, ch)

        case "polyt-data":
            dapi, polyt = input_data
            return model.eval(polyt, diameter, ch)

        case "dapi-polyt_75-25":
            dapi, polyt = input_data
            weighted_average_input = 0.25 * normalize(polyt) + 0.75 * normalize(dapi)
            return model.eval(weighted_average_input, diameter, ch)

        case "cellpose_model_D":
            dapi, polyt = input_data
            vmin, vmax = np.percentile(polyt, [2, 98])
            clip_polyt = np.clip(polyt, vmin, vmax)
            weighted_average_input = 0.25 * normalize(clip_polyt) + 0.75 * normalize(dapi)
            return model.eval(weighted_average_input, diameter, ch)

        case "dapi-polyt":
            dapi, polyt = input_data
            return model.eval(generate_mix(dapi, polyt) * 255, diameter, ch)

        case "average-z_dapi-polyt":
            dapi, polyt = input_data
            return model.eval(generate_mix(dapi, polyt) * 255, diameter, ch)

        case _:
            raise ValueError(f"unsupported format_id: {format_id!r}")


def predict_masks_from_epi_stack(
    epi_stack,
    model: CellposeModel,
    format_id: str,
    z_plane: int,
    *,
    diameter: int = 30,
    channels=(0, 0),
):
    """Run Cellpose once per FOV stack; handles average-z preprocessing when needed.

    ``diameter`` should match the ``diam_mean`` used when training / constructing the
    model (see Cellpose ``eval``): finetuned runs typically use the same value everywhere.
    """

    if format_id == AVERAGE_Z_DAPI_POLYT:
        avg_dapi = generate_z_stack(6, epi_stack)
        avg_polyt = generate_z_stack(5, epi_stack)
        pair = [avg_dapi, avg_polyt]
    else:
        dapi, polyt = slices_for_z_plane(epi_stack, z_plane)
        pair = [dapi, polyt]

    masks, _flows, _styles = cellpose_model_test_format(
        pair, model, format_id, diameter=diameter, channels=channels
    )
    return masks
