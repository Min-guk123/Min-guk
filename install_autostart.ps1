# Windows login autostart for the Excel watcher (no admin needed). Run once.
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py  = (Get-Command python).Source
$vbs = Join-Path ([Environment]::GetFolderPath("Startup")) "KosdaqRS-Watcher.vbs"
$line = 'CreateObject("Wscript.Shell").Run """' + $py + '"" ""' + $dir + '\scripts\watch_excel.py""", 0, False'
Set-Content -Path $vbs -Value $line -Encoding ASCII
Write-Host "Registered: $vbs"
Start-Process -FilePath $py -ArgumentList ('"' + $dir + '\scripts\watch_excel.py"') -WorkingDirectory $dir -WindowStyle Hidden
Write-Host "Watcher started. Saving the Excel file now triggers an update."
