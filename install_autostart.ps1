# 윈도우 로그인 때 감시 프로그램이 자동으로 켜지게 등록합니다. (한 번만 실행)
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py  = (Get-Command python).Source
$act = New-ScheduledTaskAction -Execute $py -Argument "scripts\watch_excel.py" -WorkingDirectory $dir
$trg = New-ScheduledTaskTrigger -AtLogOn
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "KosdaqRS-Watcher" -Action $act -Trigger $trg -Settings $set -Force | Out-Null
Start-ScheduledTask -TaskName "KosdaqRS-Watcher"
Write-Host "등록 완료. 지금부터 엑셀을 저장하면 자동으로 갱신됩니다."
