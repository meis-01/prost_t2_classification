param(
    [string]$ExtractDir = "D:\fastmri_prostate\T2",
    [string]$LabelsDir = "D:\fastmri_prostate\labels",
    [string]$ReconDir = "D:\fastmri_prostate\recon_t2_phase4",
    [string]$NpzDir = "D:\fastmri_prostate\npz_t2_phase4",
    [string]$RunsDir = "D:\fastmri_prostate\runs_phase4",
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int]$GradientAccumulationSteps = 4
)

$argsList = @("run", "--skip-download")

if ($ExtractDir) { $argsList += @("--extract-dir", $ExtractDir) }
if ($LabelsDir) { $argsList += @("--labels", $LabelsDir) }
if ($ReconDir) { $argsList += @("--recon-dir", $ReconDir) }
if ($NpzDir) { $argsList += @("--npz-dir", $NpzDir) }
if ($RunsDir) { $argsList += @("--runs-dir", $RunsDir) }
$argsList += @(
    "--max-coils", "4",
    "--epochs", $Epochs.ToString(),
    "--batch-size", $BatchSize.ToString(),
    "--gradient-accumulation-steps", $GradientAccumulationSteps.ToString()
)

prost-t2 @argsList
