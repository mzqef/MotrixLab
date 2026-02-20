# monitor_automl.ps1 — Section011 AutoML monitor (run on-demand)
# Usage: .\starter_kit_schedule\scripts\monitor_automl.ps1

param(
    [string]$AutomlId = "",
    [switch]$Watch,           # continuous watch mode (Ctrl+C to stop)
    [int]$IntervalSec = 120   # check interval in watch mode
)

function Show-AutomlProgress {
    # Find latest automl dir
    $logDir = "starter_kit_log"
    if ($AutomlId) {
        $automlDir = Join-Path $logDir $AutomlId
    } else {
        $automlDir = Get-ChildItem $logDir -Directory | 
            Where-Object { $_.Name -like "automl_*" -and $_.Name -notlike "*vlm*" } |
            Sort-Object Name -Descending | 
            Select-Object -First 1 -ExpandProperty FullName
    }

    $stateFile = Join-Path $automlDir "state.yaml"
    if (!(Test-Path $stateFile)) {
        Write-Host "No state.yaml found in $automlDir" -ForegroundColor Red
        return
    }

    $state = Get-Content $stateFile -Raw
    
    # Extract key fields
    $automlId   = if ($state -match 'automl_id:\s*(\S+)') { $Matches[1] } else { "?" }
    $status     = if ($state -match 'status:\s*(\S+)') { $Matches[1] } else { "?" }
    $elapsed    = if ($state -match 'elapsed_hours:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
    $budget     = if ($state -match 'budget_hours:\s*([\d.]+)') { [double]$Matches[1] } else { 8 }

    # Count trials from hp_search_history
    $trialCount = ($state | Select-String -Pattern '  trial: \d+' -AllMatches).Matches.Count

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  AutoML Monitor: $automlId"             -ForegroundColor Cyan
    Write-Host "  Status: $status | Elapsed: $([math]::Round($elapsed,2))h / ${budget}h" -ForegroundColor $(if ($status -eq 'running') {'Green'} else {'Yellow'})
    Write-Host "  Trials completed: $trialCount (budget ~$([math]::Round($budget * 60 / 30, 0)) trials @ ~30min/trial)" -ForegroundColor White
    Write-Host "========================================" -ForegroundColor Cyan

    # Parse and display each trial
    if ($trialCount -gt 0) {
        Write-Host ""
        Write-Host "  Trial  Score    WP_idx   Survival  EpLen   ForwV   ZoneA   HeightP  AliveB   LR" -ForegroundColor Yellow
        Write-Host "  -----  -----    ------   --------  -----   -----   -----   -------  ------   --" -ForegroundColor DarkGray

        # Split state into per-trial sections
        $trialBlocks = $state -split '(?=- hp_config:)' | Where-Object { $_ -match 'trial:' }
        foreach ($block in $trialBlocks) {
            $trial    = if ($block -match 'trial:\s*(\d+)') { [int]$Matches[1] } else { -1 }
            $score    = if ($block -match 'score:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $wp       = if ($block -match 'wp_idx_mean:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $termRate = if ($block -match 'termination_rate:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }
            $epLen    = if ($block -match 'episode_length_mean:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $lr       = if ($block -match 'learning_rate:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }
            $fv       = if ($block -match '\bforward_velocity:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $za       = if ($block -match '\bzone_approach:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $hp       = if ($block -match '\bheight_progress:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $ab       = if ($block -match '\balive_bonus:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
            $surv     = [math]::Round(1.0 - $termRate, 4)
            
            $scoreColor = if ($score -ge 0.4) {'Green'} elseif ($score -ge 0.3) {'Yellow'} else {'White'}
            $wpColor    = if ($wp -ge 1.0) {'Green'} elseif ($wp -ge 0.1) {'Yellow'} else {'Red'}
            
            $line = "  T{0,-4}  {1,-7:F4}  {2,-8:F3}  {3,-9:F4} {4,-7:F0}  {5,-6:F2}  {6,-6:F1}  {7,-7:F2}   {8,-7:F3}  {9:E2}" -f `
                $trial, $score, $wp, $surv, $epLen, $fv, $za, $hp, $ab, $lr
            Write-Host $line -ForegroundColor $(if ($score -ge 0.4) {'Green'} elseif ($score -ge 0.25) {'Yellow'} else {'White'})
        }

        # Show best trial
        Write-Host ""
        $bestScore = ($trialBlocks | ForEach-Object { 
            if ($_ -match 'score:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
        } | Measure-Object -Maximum).Maximum
        Write-Host "  Best score so far: $([math]::Round($bestScore,4))" -ForegroundColor Green
        
        # Estimate completion
        if ($trialCount -gt 0 -and $status -eq 'running') {
            $minutesPerTrial = ($elapsed * 60.0) / $trialCount
            $trialsRemaining = 20 - $trialCount
            $etaHours = ($trialsRemaining * $minutesPerTrial) / 60.0
            $eta = (Get-Date).AddHours($etaHours)
            Write-Host "  ~$([math]::Round($minutesPerTrial,1)) min/trial | ETA: $($eta.ToString('HH:mm')) ($([math]::Round($etaHours,1))h remaining)" -ForegroundColor DarkCyan
        }
    } else {
        Write-Host "  No trials completed yet..." -ForegroundColor DarkGray
    }
    Write-Host ""
}

if ($Watch) {
    while ($true) {
        Clear-Host
        Show-AutomlProgress
        Start-Sleep -Seconds $IntervalSec
    }
} else {
    Show-AutomlProgress
}
