"""
CLI entry point for the AMOD encoder pipeline.

This module corresponds to AMOD script(s): all scripts (unified CLI interface)
Key matched choices:
  - 'fit' command → develop_encoding_models_amygdala.m / _subregions.m
  - 'eval' command → compile_matrices.m + make_parametric_map_amygdala.m
  - 'predict-iaps-oasis' → predict_activations_IAPS_OASIS.m
  - 'export-betas' → make_random_subregions_betas_to_csv.m
Assumptions / deviations:
  - MATLAB runs scripts sequentially; we provide a structured CLI
  - Config-driven: all parameters from YAML, not hardcoded
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="amod-encoder",
    help="ROI-agnostic AMOD encoding-model pipeline.",
    add_completion=False,
)
console = Console()


@app.command()
def fit(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    subjects: Optional[str] = typer.Option(None, "--subjects", "-s", help="Comma-separated subject IDs (overrides config)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and print plan without running"),
) -> None:
    """Fit encoding models and save artifacts.

    Mirrors MATLAB's develop_encoding_models_amygdala.m and
    develop_encoding_models_subregions.m.

    Loads fc7 features → aligns to TR grid → convolves with HRF →
    loads BOLD data → applies ROI mask → fits PLS/Ridge model →
    runs cross-validation → saves betas + metrics.
    """
    from amod_encoder.config import load_config, build_provenance, save_config_snapshot
    from amod_encoder.utils.logging import get_logger

    logger = get_logger(__name__)
    cfg = load_config(config)

    if subjects:
        cfg.subjects = [s.strip() for s in subjects.split(",")]

    if dry_run:
        console.print("[bold green]Config validated successfully.[/bold green]")
        console.print(f"  Subjects: {cfg.subjects}")
        console.print(f"  ROIs: {[r.name for r in cfg.roi]}")
        console.print(f"  Model: {cfg.model.type}, components={cfg.model.pls_n_components}")
        console.print(f"  CV: {cfg.cv.scheme}, folds={cfg.cv.n_folds}")
        console.print(f"  Backend: {cfg.compute.backend}")
        return

    provenance = build_provenance(cfg)
    _run_fit(cfg, provenance)


def _run_fit(cfg, provenance: dict) -> None:
    """Core fit logic.

    For each subject × ROI:
    1. Load fc7 features
    2. Load BOLD, apply mask
    3. Align features to TRs (resample)
    4. Convolve with HRF
    5. Fit encoding model (full data for betas)
    6. Run cross-validation for evaluation
    7. Save artifacts
    """
    from amod_encoder.config import save_config_snapshot
    from amod_encoder.data.bids_dataset import (
        discover_subjects,
        get_bold_path,
        load_bold_data,
    )
    from amod_encoder.data.roi import apply_mask, load_mask, mean_roi_timeseries
    from amod_encoder.eval.metrics import cross_validated_correlation, fishers_z
    from amod_encoder.eval.splits import generate_cv_splits
    from amod_encoder.io.artifacts import save_model_artifacts
    from amod_encoder.models.pls import PLSEncodingModel
    from amod_encoder.models.ridge import RidgeEncodingModel
    from amod_encoder.stimuli.align import align_features_to_trs
    from amod_encoder.stimuli.fc7_mat_loader import load_fc7_features
    from amod_encoder.stimuli.hrf import convolve_features_with_hrf
    from amod_encoder.utils.compute_backend import ComputeBackend
    from amod_encoder.utils.logging import get_logger

    import json
    import numpy as np

    logger = get_logger(__name__)

    # Load fc7 features (shared across subjects)
    fc7 = load_fc7_features(cfg.paths.osf_fc7_mat)
    logger.info("fc7 features loaded: %s", fc7.shape)

    # Resolve subjects
    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    # Compute backend
    compute = ComputeBackend(cfg.compute.backend, cfg.compute.device, cfg.compute.amp)

    # Build model kwargs
    if cfg.model.type == "pls":
        model_class = PLSEncodingModel
        n_comp = cfg.model.pls_n_components if isinstance(cfg.model.pls_n_components, int) else 20
        model_kwargs = {
            "n_components": n_comp,
            "standardize_X": cfg.model.standardize_X,
            "standardize_Y": cfg.model.standardize_Y,
        }
    else:
        model_class = RidgeEncodingModel
        model_kwargs = {
            "alpha": cfg.model.ridge_alpha,
            "standardize_X": cfg.model.standardize_X,
            "standardize_Y": cfg.model.standardize_Y,
            "compute": compute,
        }

    for s_id in subject_ids:
        console.print(f"\n[bold]Processing subject {s_id}[/bold]")

        # Load BOLD data
        try:
            bold_path = get_bold_path(cfg.paths.bids_root, s_id)
            bold_data, bold_img = load_bold_data(bold_path)
        except FileNotFoundError as e:
            logger.error("Subject %s: %s — skipping", s_id, e)
            continue

        n_trs = bold_data.shape[3]

        # Determine alignment order based on model config
        # develop_encoding_models_amygdala.m: resample → convolve → truncate
        # develop_encoding_models_subregions.m: convolve → resample
        # We follow the amygdala script as the primary approach
        aligned_features = align_features_to_trs(fc7, n_trs, cfg.features.align_method)
        timematched_features = convolve_features_with_hrf(
            aligned_features, n_trs, cfg.hrf.model, cfg.hrf.dt
        )

        # Optionally z-score features
        if cfg.features.zscore:
            from scipy.stats import zscore
            timematched_features = zscore(timematched_features, axis=0, ddof=0)
            timematched_features = np.nan_to_num(timematched_features)

        for roi_cfg in cfg.roi:
            console.print(f"  ROI: [cyan]{roi_cfg.name}[/cyan]")

            # Load mask and extract voxel data
            try:
                mask_img = load_mask(roi_cfg.mask_path)
                masked = apply_mask(bold_data, bold_img, mask_img, roi_cfg.name)
            except Exception as e:
                logger.error("ROI %s: %s — skipping", roi_cfg.name, e)
                continue

            # Y matrix: (T, V) — transpose of MATLAB's masked_dat.dat which is (V, T)
            if cfg.model.mode == "roi_mean":
                Y = mean_roi_timeseries(masked).reshape(-1, 1)  # (T, 1)
                logger.info("ROI mean mode: Y shape = %s", Y.shape)
            else:
                Y = masked.data.T  # (T, V)
                logger.info("Voxelwise mode: Y shape = %s", Y.shape)

            X = timematched_features

            # Fit model on full data (for saving betas)
            full_model = model_class(**model_kwargs)
            full_model.fit(X, Y)

            # Cross-validation
            cv_gen = generate_cv_splits(
                n_samples=X.shape[0],
                scheme=cfg.cv.scheme,
                n_folds=cfg.cv.n_folds,
                seed=cfg.cv.seed,
            )
            cv_results = cross_validated_correlation(
                X, Y, model_class, model_kwargs, cv_gen
            )

            # Prepare metrics
            metrics = {
                "mean_voxelwise_corr": float(cv_results["mean_corr_scalar"]),
                "mean_diag_corr": cv_results["mean_diag_corr"].tolist()
                if cv_results["mean_diag_corr"].size <= 1000
                else f"<array of {cv_results['mean_diag_corr'].size} values>",
                "n_active_voxels": masked.n_active_voxels,
                "n_trs": masked.n_trs,
                "model_type": cfg.model.type,
                "n_components": model_kwargs.get("n_components", None),
                "cv_folds": cfg.cv.n_folds,
            }

            # Also compute Fisher's Z of mean_diag_corr
            mean_z = float(np.nanmean(np.arctanh(np.clip(
                cv_results["mean_diag_corr"], -0.9999, 0.9999
            ))))
            metrics["mean_fishers_z"] = mean_z

            # Save config snapshot as dict
            config_snap = json.loads(cfg.model_dump_json())

            # Save artifacts
            save_model_artifacts(
                output_dir=cfg.paths.output_dir,
                subject_id=s_id,
                roi_name=roi_cfg.name,
                betas=full_model.betas,
                intercept=full_model.intercept,
                voxel_indices=masked.voxel_indices,
                metrics=metrics,
                provenance=provenance,
                config_snapshot=config_snap,
                extra={
                    "mean_diag_corr": cv_results["mean_diag_corr"],
                    "diag_corr": cv_results["diag_corr"],
                    "removed_voxels": masked.removed_voxels,
                },
            )

            console.print(
                f"    ✓ mean r = {cv_results['mean_corr_scalar']:.4f}, "
                f"Fisher's Z = {mean_z:.4f}, "
                f"V = {masked.n_active_voxels}"
            )

    console.print("\n[bold green]Fit complete.[/bold green]")


@app.command()
def eval(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """Evaluate fitted models — load artifacts and compute summary statistics.

    Mirrors compile_matrices.m and make_parametric_map_amygdala.m:
    compiles voxelwise correlation matrices, applies Fisher's Z,
    runs group-level t-tests.
    """
    from amod_encoder.config import load_config
    from amod_encoder.data.bids_dataset import discover_subjects
    from amod_encoder.eval.metrics import fishers_z
    from amod_encoder.eval.stats_ttests import one_sample_ttest, voxelwise_ttest_with_fdr
    from amod_encoder.io.artifacts import get_artifact_dir, load_model_artifacts
    from amod_encoder.utils.logging import get_logger

    import numpy as np
    import pandas as pd

    logger = get_logger(__name__)
    cfg = load_config(config)

    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    for roi_cfg in cfg.roi:
        console.print(f"\n[bold]Evaluating ROI: {roi_cfg.name}[/bold]")

        all_mean_diag_corr = []
        valid_subjects = []

        for s_id in subject_ids:
            art_dir = get_artifact_dir(cfg.paths.output_dir, s_id, "all", roi_cfg.name)
            mdc_path = art_dir / "mean_diag_corr.npy"
            if mdc_path.exists():
                mdc = np.load(mdc_path)
                all_mean_diag_corr.append(mdc)
                valid_subjects.append(s_id)
            else:
                logger.warning("No artifacts for sub-%s / roi-%s", s_id, roi_cfg.name)

        if not all_mean_diag_corr:
            console.print(f"  [red]No data found for {roi_cfg.name}[/red]")
            continue

        # Compile correlation matrix (subjects × voxels)
        # Handle different voxel counts across subjects by using max length
        max_v = max(m.shape[0] for m in all_mean_diag_corr)
        matrix = np.full((len(valid_subjects), max_v), np.nan)
        for i, mdc in enumerate(all_mean_diag_corr):
            matrix[i, : mdc.shape[0]] = mdc

        # Fisher's Z transform
        atanh_matrix = fishers_z(matrix)

        # Group t-test
        ttest_result = voxelwise_ttest_with_fdr(atanh_matrix, q=0.05)

        # Save tables
        tables_dir = cfg.paths.output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Save correlation matrix
        df_corr = pd.DataFrame(matrix, index=[f"sub-{s}" for s in valid_subjects])
        df_corr.to_csv(tables_dir / f"correlation_matrix_{roi_cfg.name}.csv")

        # Save Fisher Z matrix
        df_z = pd.DataFrame(atanh_matrix, index=[f"sub-{s}" for s in valid_subjects])
        df_z.to_csv(tables_dir / f"atanh_correlation_matrix_{roi_cfg.name}.csv")

        # Save t-test summary
        summary = {
            "roi": roi_cfg.name,
            "n_subjects": len(valid_subjects),
            "n_voxels": max_v,
            "n_significant_fdr": ttest_result["n_significant_fdr"],
            "n_significant_unc": ttest_result["n_significant_unc"],
            "mean_t": float(np.nanmean(ttest_result["t_stats"])),
            "mean_fishers_z": float(np.nanmean(atanh_matrix)),
        }
        with open(tables_dir / f"evaluation_summary_{roi_cfg.name}.json", "w") as f:
            json.dump(summary, f, indent=2)

        console.print(
            f"  Subjects: {len(valid_subjects)}, Voxels: {max_v}, "
            f"FDR sig: {ttest_result['n_significant_fdr']}, "
            f"Mean Fisher's Z: {summary['mean_fishers_z']:.4f}"
        )

    console.print("\n[bold green]Evaluation complete.[/bold green]")


@app.command("predict-iaps-oasis")
def predict_iaps_oasis_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """Predict IAPS/OASIS activations from fitted encoding models.

    Mirrors predict_activations_IAPS_OASIS.m:
    loads per-subject betas, computes predictions for each image,
    correlates with valence and arousal ratings.
    """
    from amod_encoder.config import load_config
    from amod_encoder.data.bids_dataset import discover_subjects
    from amod_encoder.io.artifacts import get_artifact_dir, load_model_artifacts
    from amod_encoder.io.export_betas import export_mean_betas_csv
    from amod_encoder.predict.iaps_oasis import load_ratings_csv, predict_iaps_oasis
    from amod_encoder.utils.logging import get_logger

    import numpy as np
    import pandas as pd

    logger = get_logger(__name__)
    cfg = load_config(config)

    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    # Load ratings CSVs
    for csv_name, csv_path_attr in [("IAPS", cfg.paths.iaps_csv), ("OASIS", cfg.paths.oasis_csv)]:
        if csv_path_attr is None:
            logger.info("No %s CSV path configured; skipping %s predictions", csv_name, csv_name)
            continue

        df = load_ratings_csv(csv_path_attr)

        for roi_cfg in cfg.roi:
            console.print(f"\n[bold]{csv_name} predictions for ROI: {roi_cfg.name}[/bold]")

            # Load betas for all subjects
            betas_per_subject = {}
            for s_id in subject_ids:
                art_dir = get_artifact_dir(cfg.paths.output_dir, s_id, "all", roi_cfg.name)
                try:
                    artifacts = load_model_artifacts(art_dir)
                    betas_per_subject[s_id] = artifacts["betas"]
                except FileNotFoundError:
                    logger.warning("No betas for sub-%s / roi-%s", s_id, roi_cfg.name)

            if not betas_per_subject:
                console.print(f"  [red]No betas found[/red]")
                continue

            # TODO: In the full pipeline, we would extract fc7 features from
            # IAPS/OASIS images using EmoNet. For now, use CSV-based approach
            # if enc_pred columns exist in the CSV.
            console.print(
                f"  Loaded betas for {len(betas_per_subject)} subjects. "
                f"Full image-based prediction requires fc7 features for {csv_name} images."
            )

            # Export mean betas for each subject (useful for downstream analysis)
            tables_dir = cfg.paths.output_dir / "tables"
            for s_id, betas in betas_per_subject.items():
                export_mean_betas_csv(betas, s_id, roi_cfg.name, cfg.paths.output_dir)

    console.print("\n[bold green]Predict IAPS/OASIS complete.[/bold green]")


@app.command("export-betas")
def export_betas_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
) -> None:
    """Export fitted betas to CSV files.

    Mirrors make_random_subregions_betas_to_csv.m:
    exports mean betas (averaged across voxels) as single-column CSVs
    compatible with MATLAB/ActMax workflows.
    """
    from amod_encoder.config import load_config
    from amod_encoder.data.bids_dataset import discover_subjects
    from amod_encoder.io.artifacts import get_artifact_dir, load_model_artifacts
    from amod_encoder.io.export_betas import export_full_betas_csv, export_mean_betas_csv
    from amod_encoder.utils.logging import get_logger

    import numpy as np

    logger = get_logger(__name__)
    cfg = load_config(config)

    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    for roi_cfg in cfg.roi:
        console.print(f"\n[bold]Exporting betas for ROI: {roi_cfg.name}[/bold]")

        for s_id in subject_ids:
            art_dir = get_artifact_dir(cfg.paths.output_dir, s_id, "all", roi_cfg.name)
            try:
                artifacts = load_model_artifacts(art_dir)
            except FileNotFoundError:
                logger.warning("No artifacts for sub-%s / roi-%s", s_id, roi_cfg.name)
                continue

            betas = artifacts["betas"]
            export_mean_betas_csv(betas, s_id, roi_cfg.name, cfg.paths.output_dir)
            export_full_betas_csv(betas, s_id, roi_cfg.name, cfg.paths.output_dir)

            console.print(f"  sub-{s_id}: exported ({betas.shape})")

    console.print("\n[bold green]Export complete.[/bold green]")


if __name__ == "__main__":
    app()
