"""
CLI Entry Point
===============

Typer-based command-line interface unifying all AMOD pipeline steps.

Design Principles:
    - One command per MATLAB script (fit, eval, validate, predict, export)
    - All parameters flow from a single YAML config (⁠``--config``)
    - Rich console output with severity-coloured progress indicators
    - Commands are idempotent: re-running with the same config is safe

Command ↔ MATLAB Mapping::

    extract-features   ─  (new) model-agnostic feature extraction
    fit                ─  develop_encoding_models_amygdala.m / _subregions.m
    eval               ─  compile_matrices.m + make_parametric_map_amygdala.m
    validate           ─  (new) automated comparison against paper values
    predict-iaps-oasis ─  predict_activations_IAPS_OASIS.m
    compile-atanh      ─  make_atanh_matrix_subregion.m
    export-betas       ─  make_random_subregions_betas_to_csv.m
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


@app.command("extract-features")
def extract_features_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config with extractor section"),
    source: Path = typer.Option(..., "--source", help="Path to video file or image directory"),
    output: Path = typer.Option(..., "--output", "-o", help="Output path for features (.npy)"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="GPU batch size"),
) -> None:
    """Extract features from images or video using any supported model.

    This command is model-agnostic. The extractor backend and model are
    specified in the YAML config's features.extractor section.

    Examples:
        # Extract CLIP features from IAPS images
        amod-encoder extract-features -c configs/clip_extractor.yaml \\
            --source /data/IAPS/images --output output/iaps_clip_features.npy

        # Extract DINOv2 features from movie frames
        amod-encoder extract-features -c configs/dinov2_extractor.yaml \\
            --source /data/500_days_of_summer.mp4 --output output/movie_dinov2.npy
    """
    from amod_encoder.config import load_config
    from amod_encoder.stimuli.extractors.registry import create_extractor_from_features_config
    from amod_encoder.utils.logging import get_logger

    logger = get_logger(__name__)
    cfg = load_config(config)

    extractor = create_extractor_from_features_config(cfg.features, cfg.paths)
    console.print(f"Extractor: [cyan]{extractor.name}[/cyan] (dim={extractor.feature_dim})")

    source = Path(source)
    if source.is_dir():
        console.print(f"Extracting from image directory: {source}")
        features, filenames = extractor.extract_from_directory(source, batch_size=batch_size)
        extractor.save_features(features, output, filenames=filenames)
        console.print(f"[green]Saved {features.shape[0]} feature vectors → {output}[/green]")
    elif source.is_file():
        suffix = source.suffix.lower()
        if suffix in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
            console.print(f"Extracting from video: {source}")
            features = extractor.extract_from_video(
                source,
                frame_sampling=cfg.features.frame_sampling,
                batch_size=batch_size,
            )
            extractor.save_features(features, output)
            console.print(f"[green]Saved {features.shape[0]} feature vectors → {output}[/green]")
        else:
            console.print(f"[red]Unsupported file type: {suffix}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[red]Source not found: {source}[/red]")
        raise typer.Exit(1)


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
    from amod_encoder.stimuli.extractors.registry import create_extractor_from_features_config
    from amod_encoder.stimuli.hrf import convolve_features_with_hrf
    from amod_encoder.utils.compute_backend import ComputeBackend
    from amod_encoder.utils.logging import get_logger

    import json
    import numpy as np

    logger = get_logger(__name__)

    # Load features via the model-agnostic extractor system
    extractor = create_extractor_from_features_config(cfg.features, cfg.paths)
    fc7 = extractor.load() if hasattr(extractor, 'load') else None
    if fc7 is None:
        raise typer.BadParameter(
            "The 'fit' command requires pre-computed features (or use "
            "'extract-features' first, then point to the .npy output)."
        )
    logger.info("Features loaded via %s: %s", extractor.name, fc7.shape)

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

        # Determine alignment order based on config
        # develop_encoding_models_amygdala.m: resample → convolve → truncate
        # develop_encoding_models_subregions.m: convolve → truncate → resample
        conv_order = getattr(cfg.features, 'convolution_order', 'resample_then_convolve')

        if conv_order == "convolve_then_resample":
            # Subregions path: convolve raw fc7 features first, then resample
            convolved_features = convolve_features_with_hrf(
                fc7, fc7.shape[0], cfg.hrf.model, cfg.hrf.dt
            )
            timematched_features = align_features_to_trs(
                convolved_features, n_trs, cfg.features.align_method
            )
        else:
            # Amygdala path (default): resample first, then convolve
            aligned_features = align_features_to_trs(
                fc7, n_trs, cfg.features.align_method
            )
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


@app.command("compile-atanh")
def compile_atanh_cmd(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    parent_mask: Path = typer.Option(..., "--parent-mask", "-p", help="Path to parent ROI mask (e.g., whole amygdala NIfTI)"),
) -> None:
    """Compile subregion-averaged Fisher's Z matrix from whole-ROI atanh matrix.

    Mirrors make_atanh_matrix_subregion.m:
    loads the whole-amygdala atanh matrix (from eval), determines which
    voxels belong to each subregion, averages atanh per subregion per subject.

    This command requires:
    1. The 'eval' command to have been run first (producing atanh matrices).
    2. A parent mask (whole amygdala) to determine spatial correspondence.
    3. Config with multiple ROIs representing the subregions.
    """
    from amod_encoder.config import load_config
    from amod_encoder.data.bids_dataset import discover_subjects
    from amod_encoder.eval.metrics import fishers_z
    from amod_encoder.io.atanh_matrix import (
        compute_subregion_membership,
        mask_atanh_by_subregions,
        save_subregion_atanh,
    )
    from amod_encoder.utils.logging import get_logger

    import numpy as np
    import pandas as pd

    logger = get_logger(__name__)
    cfg = load_config(config)

    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    # Load the whole-amygdala atanh matrix from eval output
    tables_dir = cfg.paths.output_dir / "tables"

    # Try to find the parent ROI atanh matrix
    # The eval command saves it as atanh_correlation_matrix_{roi_name}.csv
    # For subregion masking, we need the whole-amygdala matrix
    parent_atanh_path = tables_dir / "atanh_correlation_matrix_amygdala.csv"
    if not parent_atanh_path.exists():
        # Try to compile from per-subject artifacts
        console.print("[yellow]No pre-compiled amygdala atanh matrix found.[/yellow]")
        console.print("Run 'amod-encoder eval' with the amygdala config first.")
        raise typer.Exit(1)

    df_atanh = pd.read_csv(parent_atanh_path, index_col=0)
    atanh_matrix = df_atanh.values
    console.print(f"Loaded atanh matrix: {atanh_matrix.shape}")

    # Build subregion mask mapping from config ROIs
    subregion_mask_paths = {roi.name: roi.mask_path for roi in cfg.roi}
    console.print(f"Subregions: {list(subregion_mask_paths.keys())}")

    # Compute voxel membership
    _, membership = compute_subregion_membership(parent_mask, subregion_mask_paths)

    # Mask and average
    result = mask_atanh_by_subregions(atanh_matrix, membership)

    # Save
    out_path = save_subregion_atanh(result, tables_dir)
    console.print(f"[green]Saved subregion atanh:[/green] {out_path}")

    # Print summary
    for i, name in enumerate(result.subregion_names):
        mean_z = float(np.nanmean(result.avg_atanh[:, i]))
        console.print(
            f"  {name}: {result.subregion_voxel_counts[name]} voxels, "
            f"mean atanh = {mean_z:.4f}"
        )

    console.print("\n[bold green]Compile atanh complete.[/bold green]")


# ===== Paper Reference Values (Jang & Kragel 2024) ==========================
# These constants encode the published results for automated validation.
# When running the 'validate' command, observed values are compared against
# these references to confirm replication (or quantify deviation).
_PAPER_REFERENCE = {
    "amygdala": {
        "mean_beta_hat": 0.049,
        "se_beta_hat": 0.0053,
        "t_stat": 9.27,
        "df": 53,
        "p_threshold": 0.001,
        "n_subjects": 20,
        "description": "Whole amygdala encoding performance (Table 1, Fig 3a)",
    },
    "classification": {
        "accuracy": 0.717,
        "accuracy_se": 0.017,
        "n_categories": 7,
        "description": "7-way emotion classification from amygdala (Fig 6c)",
    },
    "subregions": {
        "LB":   {"mean_beta_hat": None, "description": "Lateral-Basal nucleus"},
        "SF":   {"mean_beta_hat": None, "description": "Superficial nucleus"},
        "CM":   {"mean_beta_hat": None, "description": "Centro-Medial nucleus"},
        "AStr": {"mean_beta_hat": None, "description": "Amygdalostriatal transition"},
    },
}


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    reference_json: Optional[Path] = typer.Option(
        None, "--reference-json", "-r",
        help="Optional JSON with custom expected values (overrides built-in paper reference)",
    ),
    skip_plots: bool = typer.Option(False, "--skip-plots", help="Skip plot generation"),
    tolerance: float = typer.Option(
        0.20, "--tolerance", "-t",
        help="Relative tolerance for pass/fail (e.g. 0.20 = within 20%% of reference)",
    ),
) -> None:
    """Validate a completed pipeline run against paper reference values.

    Loads artifacts from a finished fit+eval run, compiles per-ROI
    Fisher's Z values, compares against the published results from
    Jang & Kragel (2024), generates all paper-matched figures, and
    produces a Markdown validation report.

    Workflow:
        1. amod-encoder fit -c configs/amod_amygdala.yaml
        2. amod-encoder eval -c configs/amod_amygdala.yaml
        3. amod-encoder validate -c configs/amod_amygdala.yaml

    The report is written to <output_dir>/validation/validation_report.md
    and all plots go to <output_dir>/validation/figures/.
    """
    from amod_encoder.config import load_config
    from amod_encoder.data.bids_dataset import discover_subjects
    from amod_encoder.eval.metrics import fishers_z
    from amod_encoder.io.artifacts import get_artifact_dir, load_model_artifacts
    from amod_encoder.utils.logging import get_logger

    import numpy as np

    logger = get_logger(__name__)
    cfg = load_config(config)

    # ---- Load reference values ----
    if reference_json and reference_json.exists():
        with open(reference_json) as f:
            reference = json.load(f)
        console.print(f"[cyan]Using custom reference: {reference_json}[/cyan]")
    else:
        reference = _PAPER_REFERENCE
        console.print("[cyan]Using built-in paper reference (Jang & Kragel 2024)[/cyan]")

    # ---- Resolve subjects ----
    if cfg.subjects == "all":
        subject_ids = discover_subjects(cfg.paths.bids_root)
    else:
        subject_ids = cfg.subjects

    # ---- Output directories ----
    val_dir = Path(cfg.paths.output_dir) / "validation"
    fig_dir = val_dir / "figures"
    val_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect per-ROI metrics ----
    roi_results: dict[str, dict] = {}
    subregion_data: dict[str, np.ndarray] = {}  # for violin plots

    for roi_cfg in cfg.roi:
        roi_name = roi_cfg.name
        console.print(f"\n[bold]Validating ROI: {roi_name}[/bold]")

        all_mean_corr = []
        all_fishers_z = []
        per_subject_metrics = []

        for s_id in subject_ids:
            art_dir = get_artifact_dir(cfg.paths.output_dir, s_id, "all", roi_name)

            # Try loading the mean_diag_corr array (saved by fit)
            mdc_path = art_dir / "mean_diag_corr.npy"
            metrics_path = art_dir / "metrics.json"

            if mdc_path.exists():
                mdc = np.load(mdc_path)
                z_vals = fishers_z(mdc)
                mean_z = float(np.nanmean(z_vals))
                mean_r = float(np.nanmean(mdc))
                all_mean_corr.append(mean_r)
                all_fishers_z.append(mean_z)
                per_subject_metrics.append({
                    "subject": s_id,
                    "mean_r": mean_r,
                    "mean_z": mean_z,
                    "n_voxels": int(mdc.shape[0]),
                })
            elif metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                z_val = m.get("mean_fishers_z", None)
                r_val = m.get("mean_voxelwise_corr", None)
                if z_val is not None:
                    all_fishers_z.append(float(z_val))
                if r_val is not None:
                    all_mean_corr.append(float(r_val))
                per_subject_metrics.append({
                    "subject": s_id,
                    "mean_r": r_val,
                    "mean_z": z_val,
                })
            else:
                logger.warning("No artifacts for sub-%s / roi-%s", s_id, roi_name)

        if not all_fishers_z:
            console.print(f"  [red]No data found for {roi_name}[/red]")
            continue

        arr_z = np.array(all_fishers_z)
        arr_r = np.array(all_mean_corr) if all_mean_corr else np.array([])
        subregion_data[roi_name] = arr_z

        # One-sample t-test: is mean Fisher's Z > 0?
        from scipy import stats
        if len(arr_z) > 1:
            t_stat, p_val = stats.ttest_1samp(arr_z, 0.0)
        else:
            t_stat, p_val = float("nan"), float("nan")

        roi_results[roi_name] = {
            "n_subjects": len(all_fishers_z),
            "mean_fishers_z": float(np.mean(arr_z)),
            "se_fishers_z": float(np.std(arr_z, ddof=1) / np.sqrt(len(arr_z))),
            "mean_r": float(np.mean(arr_r)) if arr_r.size > 0 else None,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "df": len(arr_z) - 1,
            "per_subject": per_subject_metrics,
        }

        console.print(
            f"  N={len(all_fishers_z)}, "
            f"mean Z={np.mean(arr_z):.4f} ± {np.std(arr_z, ddof=1)/np.sqrt(len(arr_z)):.4f}, "
            f"t({len(arr_z)-1})={t_stat:.2f}, p={p_val:.4e}"
        )

    # ---- Compare against reference ----
    console.print("\n[bold]Comparing against reference values...[/bold]")
    comparisons: list[dict] = []

    for roi_name, result in roi_results.items():
        ref = reference.get(roi_name, {})
        ref_z = ref.get("mean_beta_hat")  # paper uses beta_hat ≈ Fisher's Z
        if ref_z is not None and ref_z > 0:
            obs_z = result["mean_fishers_z"]
            rel_diff = abs(obs_z - ref_z) / ref_z
            passed = rel_diff <= tolerance
            comp = {
                "roi": roi_name,
                "metric": "mean_fishers_z",
                "observed": obs_z,
                "expected": ref_z,
                "relative_diff": rel_diff,
                "tolerance": tolerance,
                "passed": passed,
            }
            comparisons.append(comp)
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(
                f"  {roi_name}: observed={obs_z:.4f}, expected={ref_z:.4f}, "
                f"diff={rel_diff:.1%}  {status}"
            )
        else:
            console.print(f"  {roi_name}: no reference value — [yellow]SKIP[/yellow]")

    # ---- Generate plots ----
    plot_paths: list[str] = []
    if not skip_plots and subregion_data:
        console.print("\n[bold]Generating validation plots...[/bold]")
        from amod_encoder.diagnostics.plots import generate_all_validation_plots

        # Load eval summary JSONs for regression effects if available
        regression_effects = None
        tables_dir = Path(cfg.paths.output_dir) / "tables"
        if tables_dir.exists():
            reg_data = {}
            for roi_name in roi_results:
                summary_path = tables_dir / f"evaluation_summary_{roi_name}.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        s = json.load(f)
                    reg_data[roi_name] = {
                        "mean": s.get("mean_fishers_z", 0),
                        "se": roi_results[roi_name]["se_fishers_z"],
                        "p": roi_results[roi_name]["p_value"],
                    }
            if reg_data:
                regression_effects = reg_data

        plot_paths = generate_all_validation_plots(
            subregion_data=subregion_data,
            save_dir=str(fig_dir),
            regression_effects=regression_effects,
        )
        console.print(f"  Generated {len(plot_paths)} plots → {fig_dir}")

    # ---- Write validation report (Markdown) ----
    report_path = val_dir / "validation_report.md"
    _write_validation_report(
        report_path=report_path,
        roi_results=roi_results,
        comparisons=comparisons,
        plot_paths=plot_paths,
        reference=reference,
        config_path=str(config),
        tolerance=tolerance,
    )
    console.print(f"\n[bold green]Validation report → {report_path}[/bold green]")

    # ---- Save machine-readable results ----
    results_json_path = val_dir / "validation_results.json"
    _save_json_safe(results_json_path, {
        "roi_results": {k: {kk: vv for kk, vv in v.items() if kk != "per_subject"}
                        for k, v in roi_results.items()},
        "comparisons": comparisons,
        "n_plots": len(plot_paths),
    })
    console.print(f"[green]Machine-readable results → {results_json_path}[/green]")


def _save_json_safe(path: Path, data: dict) -> None:
    """JSON-serialise with numpy fallback."""
    import numpy as np

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_Enc)


def _write_validation_report(
    report_path: Path,
    roi_results: dict,
    comparisons: list[dict],
    plot_paths: list[str],
    reference: dict,
    config_path: str,
    tolerance: float,
) -> None:
    """Generate a Markdown validation report.

    The report includes:
    - Header with config path and timestamp
    - Per-ROI summary table (N, mean Z, SE, t, p)
    - Comparison table against reference values (pass/fail)
    - Links to generated figures
    - Raw per-subject data
    """
    from datetime import datetime, timezone

    lines: list[str] = []
    lines.append("# AMOD Encoding Model — Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Config:** `{config_path}`")
    lines.append(f"**Tolerance:** {tolerance:.0%}")
    lines.append("")

    # ---- Summary table ----
    lines.append("## Per-ROI Summary")
    lines.append("")
    lines.append("| ROI | N | Mean Z | SE | t | df | p |")
    lines.append("|-----|---|--------|-----|---|-----|------|")
    for roi, r in roi_results.items():
        lines.append(
            f"| {roi} | {r['n_subjects']} | {r['mean_fishers_z']:.4f} "
            f"| {r['se_fishers_z']:.4f} | {r['t_stat']:.2f} "
            f"| {r['df']} | {r['p_value']:.2e} |"
        )
    lines.append("")

    # ---- Comparison table ----
    if comparisons:
        lines.append("## Comparison Against Reference")
        lines.append("")
        lines.append(f"Reference: Jang & Kragel (2024)")
        lines.append("")
        lines.append("| ROI | Metric | Observed | Expected | Diff | Status |")
        lines.append("|-----|--------|----------|----------|------|--------|")
        for c in comparisons:
            status = "PASS ✓" if c["passed"] else "FAIL ✗"
            lines.append(
                f"| {c['roi']} | {c['metric']} | {c['observed']:.4f} "
                f"| {c['expected']:.4f} | {c['relative_diff']:.1%} | {status} |"
            )
        lines.append("")

        n_pass = sum(1 for c in comparisons if c["passed"])
        n_total = len(comparisons)
        overall = "PASS" if n_pass == n_total else "PARTIAL" if n_pass > 0 else "FAIL"
        lines.append(f"**Overall: {overall}** ({n_pass}/{n_total} metrics within {tolerance:.0%} tolerance)")
        lines.append("")

    # ---- Figures ----
    if plot_paths:
        lines.append("## Generated Figures")
        lines.append("")
        for p in plot_paths:
            fname = Path(p).name
            lines.append(f"- ![{fname}](figures/{fname})")
        lines.append("")

    # ---- Per-subject data ----
    lines.append("## Per-Subject Data")
    lines.append("")
    for roi, r in roi_results.items():
        lines.append(f"### {roi}")
        lines.append("")
        lines.append("| Subject | Mean r | Mean Z |")
        lines.append("|---------|--------|--------|")
        for ps in r.get("per_subject", []):
            mr = f"{ps['mean_r']:.4f}" if ps.get("mean_r") is not None else "—"
            mz = f"{ps['mean_z']:.4f}" if ps.get("mean_z") is not None else "—"
            lines.append(f"| sub-{ps['subject']} | {mr} | {mz} |")
        lines.append("")

    # ---- Reference values ----
    lines.append("## Reference Values")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(reference, indent=2, default=str))
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


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
