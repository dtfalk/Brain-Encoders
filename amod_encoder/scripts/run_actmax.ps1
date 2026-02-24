<#
.SYNOPSIS
    Runs ActMax activation maximization for all subjects × ROIs.
    Windows equivalent of AMOD-main/scripts/generate_artificial_stim.sh

.DESCRIPTION
    Prerequisites:
      1. All four replication configs have been run with amod-encoder fit + export-betas
      2. ActMax repo cloned: git clone https://github.com/Animadversio/ActMax-Optimizer-Dev data\ActMax
      3. ActMax conda env set up (see data\ActMax\environment.yml)
      4. DGN + EmoNet weights placed in data\ActMax\weights\ (get from OSF or authors)

    Mean betas CSVs are expected at:
      results\{roi}\tables\meanbeta_sub-{s}_{roi}_fc7_invert_imageFeatures.csv

    Output PNGs land at:
      results\artificial_stim\{roi}\emonet_fc7_sub{s}_{roi}_run*.png

.EXAMPLE
    # Run from Brain-Encoders\amod_encoder\ directory
    cd C:\Users\David\Desktop\code\Brain-Encoders\amod_encoder
    .\scripts\run_actmax.ps1
#>

$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
$ACTMAX_DIR  = "$PSScriptRoot\..\data\ActMax"   # path to cloned ActMax repo
$RESULTS_DIR = "$PSScriptRoot\..\results"
$OUTPUT_DIR  = "$PSScriptRoot\..\results\artificial_stim"
$CONDA_ENV   = "AMOD"                            # conda env with ActMax deps

# ROI → subdirectory name in results\ + ActMax output subfolder
$ROI_MAP = [ordered]@{
    "amygdala"     = "amygdala"       # whole amygdala
    "AStr"         = "subregions"     # subregion betas are under results\subregions\
    "CM"           = "subregions"
    "LB"           = "subregions"
    "SF"           = "subregions"
    "visualcortex" = "visual_cortex"  # note: export uses 'visual_cortex', ActMax uses 'visualcortex'
    "itcortex"     = "inferotemporal"
}

$SUBJECTS = 1..20

# --------------------------------------------------------------------------
# Helper: get the CSV path for a given subject + roi
# --------------------------------------------------------------------------
function Get-BetaCSV {
    param($Subject, $Roi, $ResultsSubdir)
    $padded = "{0:D2}" -f $Subject
    $csv = Join-Path $RESULTS_DIR "$ResultsSubdir\tables\meanbeta_sub-$padded`_${Roi}_fc7_invert_imageFeatures.csv"
    return $csv
}

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
foreach ($roi_key in $ROI_MAP.Keys) {
    $results_subdir = $ROI_MAP[$roi_key]
    $out_folder = Join-Path $OUTPUT_DIR $roi_key

    foreach ($sub in $SUBJECTS) {
        $padded = "{0:D2}" -f $sub
        $csv = Get-BetaCSV -Subject $sub -Roi $roi_key -ResultsSubdir $results_subdir

        if (-not (Test-Path $csv)) {
            Write-Warning "Missing betas CSV for sub-$padded / $roi_key — skipping ($csv)"
            continue
        }

        $out_filename = "emonet_fc7_sub${padded}_${roi_key}_randinit"

        Write-Host "[$roi_key] sub-$padded → $out_filename" -ForegroundColor Cyan

        # Activate conda env and run ActMax
        # The conda run approach avoids needing to activate in PS first
        conda run -n $CONDA_ENV python "$ACTMAX_DIR\ActMax.py" `
            --act_layer ".Conv2dConv_6" `
            --encoding_filename $csv `
            --output_filename $out_filename `
            --output_folder $out_folder

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "ActMax failed for sub-$padded / $roi_key (exit $LASTEXITCODE) — continuing"
        }
    }
}

Write-Host "`nDone. Generated images in: $OUTPUT_DIR" -ForegroundColor Green
