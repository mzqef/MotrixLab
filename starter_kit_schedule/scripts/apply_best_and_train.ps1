# apply_best_and_train.ps1 — Section011 post-AutoML: apply best config to cfg.py + rl_cfgs.py, then launch full 100M training
# Usage: .\starter_kit_schedule\scripts\apply_best_and_train.ps1 [-AutomlId automl_20260219_043704] [-DryRun]
#
# Steps:
#   1. Read AutoML state.yaml → extract best trial config
#   2. Show summary and ask confirmation
#   3. Apply reward_scales to cfg.py BASE_REWARD_SCALES
#   4. Apply rl_overrides (learning_rate, entropy_loss_scale) to rl_cfgs.py
#   5. Launch: uv run scripts/train.py --env vbot_navigation_section011

param(
    [string]$AutomlId = "",
    [switch]$DryRun,
    [switch]$NoConfirm
)

Set-Location d:\MotrixLab

# ─── Find AutoML state ──────────────────────────────────────────────────────
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
    Write-Host "ERROR: No state.yaml in $automlDir" -ForegroundColor Red
    exit 1
}

$state = Get-Content $stateFile -Raw
$automlIdName = if ($state -match 'automl_id:\s*(\S+)') { $Matches[1] } else { "unknown" }
$status = if ($state -match 'status:\s*(\S+)') { $Matches[1] } else { "?" }

Write-Host ""
Write-Host "AutoML: $automlIdName (status: $status)" -ForegroundColor Cyan

# ─── Parse trial blocks ─────────────────────────────────────────────────────
$trialBlocks = $state -split '(?=- hp_config:)' | Where-Object { $_ -match 'trial:' }
if ($trialBlocks.Count -eq 0) {
    Write-Host "ERROR: No completed trials found" -ForegroundColor Red
    exit 1
}

$bestScore = -1
$bestBlock = $null
foreach ($block in $trialBlocks) {
    $score = if ($block -match 'score:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
    if ($score -gt $bestScore) {
        $bestScore = $score
        $bestBlock = $block
    }
}

$bestTrial  = if ($bestBlock -match 'trial:\s*(\d+)') { [int]$Matches[1] } else { -1 }
$bestWP     = if ($bestBlock -match 'wp_idx_mean:\s*([\d.]+)') { [double]$Matches[1] } else { 0 }
$bestSurv   = if ($bestBlock -match 'termination_rate:\s*([\d.eE+\-]+)') { 1.0 - [double]$Matches[1] } else { 0 }
$bestLR     = if ($bestBlock -match 'learning_rate:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }
$bestEnt    = if ($bestBlock -match 'entropy_loss_scale:\s*([\d.eE+\-]+)') { [double]$Matches[1] } else { 0 }

Write-Host ""
Write-Host "Best trial: T$bestTrial  score=$([math]::Round($bestScore,4))  wp_idx=$([math]::Round($bestWP,3))  survival=$([math]::Round($bestSurv,4))" -ForegroundColor Green
Write-Host "  LR=$bestLR  entropy=$bestEnt" -ForegroundColor Green

# ─── Extract reward_config block ────────────────────────────────────────────
# Parse the reward_config YAML sub-block from the best trial
$rewardLines = @{}
$inRewardBlock = $false
foreach ($line in ($bestBlock -split "`n")) {
    if ($line -match '^\s+reward_config:') {
        $inRewardBlock = $true
        continue
    }
    if ($inRewardBlock) {
        if ($line -match '^\s{4}(\w+):\s*([\-\d.eE+]+)') {
            $rewardLines[$Matches[1].Trim()] = $Matches[2].Trim()
        } elseif ($line -match '^\s{2}\w' -or $line -match '^\s*$') {
            $inRewardBlock = $false
        }
    }
}

if ($rewardLines.Count -eq 0) {
    Write-Host "ERROR: Could not parse reward_config from best trial" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Reward config to apply ($($rewardLines.Count) keys):" -ForegroundColor Yellow
foreach ($kv in ($rewardLines.GetEnumerator() | Sort-Object Key)) {
    Write-Host "  $($kv.Key): $($kv.Value)" -ForegroundColor DarkCyan
}

# ─── Dry run exit ────────────────────────────────────────────────────────────
if ($DryRun) {
    Write-Host ""
    Write-Host "[DRY RUN] No changes applied." -ForegroundColor DarkYellow
    exit 0
}

# ─── Confirmation ────────────────────────────────────────────────────────────
if (!$NoConfirm) {
    Write-Host ""
    $ans = Read-Host "Apply these configs to cfg.py + rl_cfgs.py and launch 100M training? [y/N]"
    if ($ans -notmatch '^[Yy]') {
        Write-Host "Aborted." -ForegroundColor Yellow
        exit 0
    }
}

# ─── Apply to cfg.py using Python ────────────────────────────────────────────
Write-Host ""
Write-Host "Applying reward scales to cfg.py..." -ForegroundColor Cyan

$cfgPath = "starter_kit/navigation2/vbot/cfg.py"
$cfgContent = Get-Content $cfgPath -Raw

# Update each reward scale key in BASE_REWARD_SCALES
foreach ($kv in $rewardLines.GetEnumerator()) {
    $key = $kv.Key
    $val = $kv.Value
    # Match patterns like:   "key": 123.45,  or  "key": -123.45,
    $pattern = '("' + [regex]::Escape($key) + '":\s*)[\-\d.eE+]+'
    if ($cfgContent -match $pattern) {
        $cfgContent = $cfgContent -replace $pattern, "`${1}$val"
        Write-Host "  Updated: $key = $val" -ForegroundColor Green
    } else {
        Write-Host "  SKIP (not found): $key" -ForegroundColor DarkYellow
    }
}

Set-Content $cfgPath $cfgContent -NoNewline
Write-Host "cfg.py updated." -ForegroundColor Green

# ─── Apply to rl_cfgs.py ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "Applying RL config to rl_cfgs.py..." -ForegroundColor Cyan

$rlPath = "starter_kit/navigation2/vbot/rl_cfgs.py"
$rlContent = Get-Content $rlPath -Raw

# learning_rate
if ($bestLR -gt 0) {
    $rlContent = $rlContent -replace '(learning_rate:\s*float\s*=\s*)[\d.eE+\-]+', "`${1}$bestLR"
    Write-Host "  learning_rate = $bestLR" -ForegroundColor Green
}
# entropy_loss_scale  
if ($bestEnt -gt 0) {
    $rlContent = $rlContent -replace '(entropy_loss_scale:\s*float\s*=\s*)[\d.eE+\-]+', "`${1}$bestEnt"
    Write-Host "  entropy_loss_scale = $bestEnt" -ForegroundColor Green
}

Set-Content $rlPath $rlContent -NoNewline
Write-Host "rl_cfgs.py updated." -ForegroundColor Green

# ─── Launch full 100M training ───────────────────────────────────────────────
Write-Host ""
Write-Host "Launching 100M training for vbot_navigation_section011..." -ForegroundColor Cyan
Write-Host "  Run: uv run scripts/train.py --env vbot_navigation_section011" -ForegroundColor White
Write-Host ""

uv run scripts/train.py --env vbot_navigation_section011
