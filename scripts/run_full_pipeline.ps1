param(
    [string]$ExtractDir = "D:\fastmri_prostate\T2",
    [string]$LabelsDir = "D:\fastmri_prostate\labels",
    [string]$ReconDir = "D:\fastmri_prostate\recon_t2",
    [string]$NpzDir = "D:\fastmri_prostate\npz_t2_middle_coil",
    [string]$RunsDir = "D:\fastmri_prostate\runs",
    [int]$Epochs = 20,
    [int]$BatchSize = 32
)

$argsList = @("run", "--skip-download")

if ($ExtractDir) { $argsList += @("--extract-dir", $ExtractDir) }
if ($LabelsDir) { $argsList += @("--labels", $LabelsDir) }
if ($ReconDir) { $argsList += @("--recon-dir", $ReconDir) }
if ($NpzDir) { $argsList += @("--npz-dir", $NpzDir) }
if ($RunsDir) { $argsList += @("--runs-dir", $RunsDir) }
$argsList += @("--max-coils", "1", "--epochs", $Epochs.ToString(), "--batch-size", $BatchSize.ToString())

prost-t2 @argsList
