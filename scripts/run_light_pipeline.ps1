param(
    [string]$DownloadScript = ".\prostate_download_script.txt",
    [string]$DownloadDir,
    [string]$ExtractDir,
    [string]$ReconDir,
    [string]$NpzDir,
    [string]$RunsDir,
    [string]$LightCounts,
    [int]$Epochs = 1,
    [int]$BatchSize = 4,
    [switch]$SkipTrain
)

$argsList = @("run", "--light", "--download-script", $DownloadScript)

if ($DownloadDir) { $argsList += @("--download-dir", $DownloadDir) }
if ($ExtractDir) { $argsList += @("--extract-dir", $ExtractDir) }
if ($ReconDir) { $argsList += @("--recon-dir", $ReconDir) }
if ($NpzDir) { $argsList += @("--npz-dir", $NpzDir) }
if ($RunsDir) { $argsList += @("--runs-dir", $RunsDir) }
if ($LightCounts) { $argsList += @("--light-counts", $LightCounts) }

if ($SkipTrain) {
    $argsList += "--skip-train"
} else {
    $argsList += @("--epochs", $Epochs.ToString(), "--batch-size", $BatchSize.ToString())
}

prost-t2 @argsList
