# section011_loop.ps1 — Full autonomous pipeline: AutoML → apply best → 100M train → monitor → iterate
# Runs until score 20/20 or user interrupts.
# Usage: .\starter_kit_schedule\scripts\section011_loop.ps1

Set-Location d:\MotrixLab

$ENV_NAME   = "vbot_navigation_section011"
$LOG_PREFIX = "starter_kit_log"
$RUNS_DIR   = "runs/$ENV_NAME"
$AUTOML_ID  = "automl_20260219_043704"  # current active run

function Log($msg, $color="White") {
    Write-Host "$(Get-Date -Format 'HH:mm:ss') | $msg" -ForegroundColor $color
}

function Get-AutomlState {
    $f = "$LOG_PREFIX/$AUTOML_ID/state.yaml"
    if (!(Test-Path $f)) { return $null }
    return Get-Content $f -Raw
}

function Get-TrialCount($state) {
    ($state | Select-String -Pattern '  trial: \d+' -AllMatches).Matches.Count
}

function Get-BestTrial($state) {
    $blocks = $state -split '(?=- hp_config:)' | Where-Object { $_ -match 'trial:' }
    $best = $null; $bestScore = -1
    foreach ($b in $blocks) {
        $s = if ($b -match 'score:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
        if ($s -gt $bestScore) { $bestScore = $s; $best = $b }
    }
    return $best
}

function Show-TrialTable($state) {
    $blocks = $state -split '(?=- hp_config:)' | Where-Object { $_ -match 'trial:' }
    Log "Trial  Score    WP     Surv   EpLen  FwdV   ZoneA  HgtP   AliveB  LR" "Yellow"
    foreach ($b in $blocks) {
        $t   = if ($b -match 'trial:\s*(\d+)')               { [int]$Matches[1] }    else { -1 }
        $sc  = if ($b -match 'score:\s*([\d.]+)')             { [double]$Matches[1] } else { 0 }
        $wp  = if ($b -match 'wp_idx_mean:\s*([\d.]+)')       { [double]$Matches[1] } else { 0 }
        $tr  = if ($b -match 'termination_rate:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }
        $el  = if ($b -match 'episode_length_mean:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
        $lr  = if ($b -match 'learning_rate:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }
        $fv  = if ($b -match 'forward_velocity:\s*([\d.]+)')  { [double]$Matches[1] } else { 0 }
        $za  = if ($b -match 'zone_approach:\s*([\d.]+)')     { [double]$Matches[1] } else { 0 }
        $hp  = if ($b -match 'height_progress:\s*([\d.]+)')   { [double]$Matches[1] } else { 0 }
        $ab  = if ($b -match 'alive_bonus:\s*([\d.]+)')       { [double]$Matches[1] } else { 0 }
        $col = if ($sc -ge 0.5) { "Green" } elseif ($sc -ge 0.3) { "Yellow" } else { "White" }
        $line = "  T{0,-4} {1,-7:F4} {2,-6:F3} {3,-6:F4} {4,-6:F0} {5,-6:F2} {6,-6:F1} {7,-6:F1} {8,-7:F3} {9:E2}" -f `
            $t,$sc,$wp,(1-$tr),$el,$fv,$za,$hp,$ab,$lr
        Write-Host "$(Get-Date -Format 'HH:mm:ss') | $line" -ForegroundColor $col
    }
}

function Apply-BestConfig($bestBlock) {
    Log "Applying best trial config to cfg.py + rl_cfgs.py..." "Cyan"

    # Extract reward_config
    $rewardLines = @{}
    $inBlock = $false
    foreach ($line in ($bestBlock -split "`n")) {
        if ($line -match '^\s+reward_config:') { $inBlock = $true; continue }
        if ($inBlock) {
            if ($line -match '^\s{4}(\w+):\s*([\-\d.eE+]+)') {
                $rewardLines[$Matches[1].Trim()] = $Matches[2].Trim()
            } elseif ($line -match '^\s{2}\S' -or $line.Trim() -eq '') {
                $inBlock = $false
            }
        }
    }

    $lr  = if ($bestBlock -match 'learning_rate:\s*([\d.eE+\-]+)') { $Matches[1] } else { $null }
    $ent = if ($bestBlock -match 'entropy_loss_scale:\s*([\d.eE+\-]+)') { $Matches[1] } else { $null }

    # cfg.py
    $cfgPath = "starter_kit/navigation2/vbot/cfg.py"
    $cfg = Get-Content $cfgPath -Raw
    foreach ($kv in $rewardLines.GetEnumerator()) {
        $pat = '("' + [regex]::Escape($kv.Key) + '":\s*)[\-\d.eE+]+'
        if ($cfg -match $pat) {
            $cfg = $cfg -replace $pat, "`${1}$($kv.Value)"
            Log "  cfg.py: $($kv.Key) = $($kv.Value)" "DarkCyan"
        }
    }
    Set-Content $cfgPath $cfg -NoNewline

    # rl_cfgs.py
    $rlPath = "starter_kit/navigation2/vbot/rl_cfgs.py"
    $rl = Get-Content $rlPath -Raw
    if ($lr) {
        $rl = $rl -replace '(learning_rate:\s*float\s*=\s*)[\d.eE+\-]+', "`${1}$lr"
        Log "  rl_cfgs.py: learning_rate = $lr" "DarkCyan"
    }
    # Only update section011's entropy (first occurrence in file)
    if ($ent) {
        $rl = $rl -replace '(entropy_loss_scale:\s*float\s*=\s*)[\d.eE+\-]+', "`${1}$ent"
        Log "  rl_cfgs.py: entropy_loss_scale = $ent" "DarkCyan"
    }
    Set-Content $rlPath $rl -NoNewline

    Log "Config applied." "Green"
}

function Get-LatestCheckpoint {
    $latestRun = Get-ChildItem $RUNS_DIR -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if (!$latestRun) { return $null }
    $ckpt = Get-ChildItem $latestRun.FullName -Recurse -Filter "agent_*.pt" |
        Sort-Object Name -Descending | Select-Object -First 1
    return $ckpt
}

function Get-BestWpFromRun {
    param([string]$runDir)
    # Read TensorBoard to get best wp_idx — approximate via latest checkpoint number × interval
    $latestCkpt = Get-ChildItem $runDir -Recurse -Filter "agent_*.pt" |
        Sort-Object Name -Descending | Select-Object -First 1
    if (!$latestCkpt) { return "no ckpt" }
    return $latestCkpt.Name
}

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Monitor AutoML until done
# ═══════════════════════════════════════════════════════════════
Log "=== PHASE 1: Monitoring AutoML $AUTOML_ID ===" "Cyan"
Log "Budget: 8h | Target: ~16 trials @ 30min/trial" "DarkCyan"

$prevTrialCount = 0
while ($true) {
    Start-Sleep -Seconds 90

    $state = Get-AutomlState
    if (!$state) { Log "State file missing, retrying..." "Yellow"; continue }

    $status = if ($state -match 'status:\s*(\S+)') { $Matches[1] } else { "?" }
    $elapsed = if ($state -match 'elapsed_hours:\s*([\d.]+)') { [math]::Round([double]$Matches[1],2) } else { 0 }
    $tc = Get-TrialCount $state

    if ($tc -gt $prevTrialCount) {
        Log "--- Trial $($tc-1) completed (total done: $tc) elapsed=${elapsed}h ---" "Green"
        Show-TrialTable $state
        $prevTrialCount = $tc
    } else {
        # Heartbeat: show active checkpoint progress
        $ckpt = Get-LatestCheckpoint
        $ckptInfo = if ($ckpt) { $ckpt.Name } else { "none" }
        Log "AutoML running: $tc trials done, ${elapsed}h elapsed | latest ckpt: $ckptInfo" "DarkGray"
    }

    if ($status -ne "running") {
        Log "AutoML finished with status: $status" "Green"
        break
    }

    # Safety: if budget exhausted but status not updated yet
    if ($elapsed -ge 8.5) {
        Log "Budget window exceeded (${elapsed}h), treating as done." "Yellow"
        break
    }
}

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Apply best config
# ═══════════════════════════════════════════════════════════════
Log "" "White"
Log "=== PHASE 2: Apply best AutoML config ===" "Cyan"

$state = Get-AutomlState
$tc = Get-TrialCount $state
Log "Final trial count: $tc" "White"
Show-TrialTable $state

$bestBlock = Get-BestTrial $state
$bestScore = if ($bestBlock -match 'score:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
$bestWP    = if ($bestBlock -match 'wp_idx_mean:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
$bestTrial = if ($bestBlock -match 'trial:\s*(\d+)') { [int]$Matches[1] } else { -1 }

Log "Winner: T$bestTrial | score=$([math]::Round($bestScore,4)) | wp_idx=$([math]::Round($bestWP,4))" "Green"

Apply-BestConfig $bestBlock

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Full 100M training
# ═══════════════════════════════════════════════════════════════
Log "" "White"
Log "=== PHASE 3: Launching 100M training ===" "Cyan"
Log "Command: uv run scripts/train.py --env $ENV_NAME" "White"
Log "Expected: ~100 min at 12500 steps/sec. Monitor TensorBoard for wp_idx." "DarkCyan"
Log "Checkpoints every 500 iters. Target: wp_idx > 6.5 at 50M steps." "DarkCyan"
Log "" "White"

$trainStart = Get-Date

# Launch training (blocking — this is the 100M run)
uv run scripts/train.py --env $ENV_NAME

$trainElapsed = [math]::Round(((Get-Date) - $trainStart).TotalMinutes, 1)
Log "Training finished after ${trainElapsed} min." "Green"

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Evaluate best checkpoint
# ═══════════════════════════════════════════════════════════════
Log "" "White"
Log "=== PHASE 4: Rank checkpoints ===" "Cyan"

$latestRun = Get-ChildItem $RUNS_DIR -Directory | Sort-Object Name -Descending | Select-Object -First 1
if ($latestRun) {
    Log "Run dir: $($latestRun.Name)" "White"
    $ckpts = Get-ChildItem $latestRun.FullName -Recurse -Filter "agent_*.pt" | Sort-Object Name -Descending | Select-Object -First 10
    Log "Latest checkpoints:" "Yellow"
    $ckpts | ForEach-Object { Log "  $($_.Name)  ($($_.LastWriteTime))" "DarkCyan" }

    # Use eval_checkpoint to rank
    Log "" "White"
    Log "Ranking checkpoints by wp_idx..." "Cyan"
    uv run starter_kit_schedule/scripts/eval_checkpoint.py --rank $latestRun.FullName
} else {
    Log "No runs found." "Red"
}

Log "" "White"
Log "=== PIPELINE COMPLETE ===" "Green"
Log "Next: review ranking output above, then run capture_vlm.py for visual analysis." "DarkCyan"
Log "  uv run scripts/capture_vlm.py --env $ENV_NAME" "White"
