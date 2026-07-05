param(
    [string]$DownloadScript = ".\prostate_download_script.txt",
    [string]$DownloadDir,
    [string]$ExtractDir,
    [string]$ReconDir,
    [string]$NpzDir,
    [string]$RunsDir
)

$argsList = @("run", "--download-script", $DownloadScript)

if ($DownloadDir) { $argsList += @("--download-dir", $DownloadDir) }
if ($ExtractDir) { $argsList += @("--extract-dir", $ExtractDir) }
if ($ReconDir) { $argsList += @("--recon-dir", $ReconDir) }
if ($NpzDir) { $argsList += @("--npz-dir", $NpzDir) }
if ($RunsDir) { $argsList += @("--runs-dir", $RunsDir) }

prost-t2 @argsList
