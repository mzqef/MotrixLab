param([int]$IntervalSec = 900, [int]$MaxChecks = 80)
Set-Location d:\MotrixLab
$maxSteps = 34500

function Get-Latest {
    param($tag, $lines)
    $line = ($lines | Select-String $tag | Select-Object -First 1).Line
    if ($line -match '0\.0000\s+([\d.]+)') { return [double]$Matches[1] }
    return 0
}

$runDir = (Get-ChildItem runs/vbot_navigation_section011/ -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$startTime = (Get-Item $runDir).CreationTime
Write-Host "=== Section011 Monitor ===" -ForegroundColor Cyan
Write-Host "Run: $(Split-Path $runDir -Leaf)" -ForegroundColor Cyan
Write-Host "[time] iter/34.5k(%)  ETA  | Term% | WP    | Smiley | RedPkt | Celeb" -ForegroundColor DarkGray

for ($i = 1; $i -le $MaxChecks; $i++) {
    $ts = Get-Date -Format "HH:mm"

    $latest = Get-ChildItem $runDir -Recurse |
        Where-Object { $_.Name -match "agent_\d+\.pt" } |
        Sort-Object { [int]($_.Name -replace 'agent_(\d+)\.pt', '$1') } -Descending |
        Select-Object -First 1
    $iter    = [int]($latest.Name -replace 'agent_(\d+)\.pt', '$1')
    $pct     = [math]::Round($iter * 100.0 / $maxSteps, 1)
    $elapsed = ((Get-Date) - $startTime).TotalMinutes
    $etaMin  = ($maxSteps - $iter) / ($iter / [math]::Max($elapsed, 1))
    $eta     = (Get-Date).AddMinutes($etaMin)

    $mon    = uv run starter_kit_schedule/scripts/monitor_training.py --env vbot_navigation_section011 2>&1
    $wp     = Get-Latest "WP Index" $mon
    $rp     = Get-Latest "Red Packet Bonus" $mon
    $cb     = Get-Latest "Celeb Bonus" $mon
    $sm     = Get-Latest "Smiley Bonus" $mon
    $tr     = if (($mon | Select-String "Termination rate") -match '([\d.]+)%') { [double]$Matches[1] } else { 100 }

    $flag = ""
    if ($cb -gt 1.0)   { $flag = " <<< CELEBRATION!" }
    elseif ($rp -gt 1.0) { $flag = " <<< RED PACKETS!" }
    elseif ($tr -lt 50) { $flag = " < term crossed 50%" }

    $termColor = if ($tr -lt 40) { "Green" } elseif ($tr -lt 60) { "Yellow" } else { "Red" }
    $rpColor   = if ($rp -gt 1.0) { "Green" } else { "DarkGray" }
    $wpColor   = if ($wp -ge 3.0) { "Green" } elseif ($wp -ge 1.0) { "Yellow" } else { "White" }

    Write-Host "[$ts] " -NoNewline -ForegroundColor DarkCyan
    Write-Host "$iter/34.5k ($pct%) ETA:$($eta.ToString('HH:mm'))" -NoNewline
    Write-Host " | Term:" -NoNewline
    Write-Host "$([math]::Round($tr,0))%" -NoNewline -ForegroundColor $termColor
    Write-Host " | WP:" -NoNewline
    Write-Host "$([math]::Round($wp,3))" -NoNewline -ForegroundColor $wpColor
    Write-Host " | Smiley:$([math]::Round($sm,1))" -NoNewline
    Write-Host " | RedPkt:" -NoNewline
    Write-Host "$([math]::Round($rp,2))" -NoNewline -ForegroundColor $rpColor
    Write-Host " | Celeb:$([math]::Round($cb,2))$flag"

    if ($iter -ge ($maxSteps - 500)) {
        Write-Host ""
        Write-Host "=== TRAINING COMPLETE at iter=$iter ===" -ForegroundColor Green
        Write-Host "Next: uv run scripts/play.py --env vbot_navigation_section011" -ForegroundColor Yellow
        break
    }

    Start-Sleep -Seconds $IntervalSec
}
