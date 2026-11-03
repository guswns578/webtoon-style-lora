param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetDir,

    [Parameter(Mandatory = $true)]
    [string]$SdScriptsDir,

    [Parameter(Mandatory = $true)]
    [string]$PretrainedModel,

    [string]$OutputDir = "",
    [string]$OutputName = "locon_lora",
    [int]$Resolution = 512,
    [int]$Repeats = 10,
    [int]$Epochs = 10,
    [int]$BatchSize = 1,
    [int]$NetworkDim = 32,
    [int]$NetworkAlpha = 16,
    [int]$ConvDim = 16,
    [int]$ConvAlpha = 8,
    [double]$UnetLr = 0.0001,
    [double]$TextEncoderLr = 0.00001,
    [int]$Seed = 42,
    [switch]$Sdxl,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $scriptRoot "output"
}

function Resolve-RequiredPath {
    param(
        [string]$Path,
        [string]$Label,
        [switch]$Leaf
    )

    if ($Leaf) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "$Label file was not found: $Path"
        }
    }
    else {
        if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
            throw "$Label directory was not found: $Path"
        }
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Convert-ToTomlPath {
    param([string]$Path)
    return (($Path -replace "\\", "/") -replace '"', '\"')
}

$datasetPath = Resolve-RequiredPath $DatasetDir "Dataset"
$sdScriptsPath = Resolve-RequiredPath $SdScriptsDir "sd-scripts"
$modelPath = Resolve-RequiredPath $PretrainedModel "Pretrained model" -Leaf

$trainScriptName = if ($Sdxl) { "sdxl_train_network.py" } else { "train_network.py" }
$trainScriptPath = Join-Path $sdScriptsPath $trainScriptName
if (-not (Test-Path -LiteralPath $trainScriptPath -PathType Leaf)) {
    throw "$trainScriptName was not found: $trainScriptPath"
}

$imageExtensions = @(".jpg", ".jpeg", ".png", ".webp", ".bmp")
$images = @(Get-ChildItem -LiteralPath $datasetPath -File |
    Where-Object { $imageExtensions -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object Name)

if ($images.Count -eq 0) {
    throw "No training images were found in the dataset directory: $datasetPath"
}

$missingCaptions = @()
foreach ($image in $images) {
    $caption = Join-Path $image.DirectoryName ($image.BaseName + ".txt")
    if (-not (Test-Path -LiteralPath $caption -PathType Leaf)) {
        $missingCaptions += $image.Name
    }
}

if ($missingCaptions.Count -gt 0) {
    $sample = ($missingCaptions | Select-Object -First 10) -join ", "
    throw "$($missingCaptions.Count) image(s) are missing matching .txt captions. Examples: $sample"
}

$orphanCaptions = @(Get-ChildItem -LiteralPath $datasetPath -File -Filter "*.txt" |
    Where-Object {
        $captionFile = $_
        $base = $captionFile.BaseName
        -not ($imageExtensions | Where-Object {
            Test-Path -LiteralPath (Join-Path $captionFile.DirectoryName ($base + $_)) -PathType Leaf
        })
    })

if ($orphanCaptions.Count -gt 0) {
    Write-Warning "$($orphanCaptions.Count) caption file(s) have no matching image. They will not be used."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$logDir = Join-Path $scriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$datasetConfigPath = Join-Path $scriptRoot "dataset_locon.generated.toml"
$datasetToml = @"
[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = [$Resolution, $Resolution]
batch_size = $BatchSize
enable_bucket = true
bucket_reso_steps = 64
bucket_no_upscale = false

[[datasets.subsets]]
image_dir = "$(Convert-ToTomlPath $datasetPath)"
num_repeats = $Repeats
"@
Set-Content -LiteralPath $datasetConfigPath -Value $datasetToml -Encoding UTF8

$commonArgs = @(
    "launch",
    "--num_cpu_threads_per_process=4",
    $trainScriptPath,
    "--pretrained_model_name_or_path=$modelPath",
    "--dataset_config=$datasetConfigPath",
    "--output_dir=$OutputDir",
    "--output_name=$OutputName",
    "--logging_dir=$logDir",
    "--save_model_as=safetensors",
    "--save_precision=fp16",
    "--mixed_precision=fp16",
    "--max_train_epochs=$Epochs",
    "--learning_rate=$UnetLr",
    "--unet_lr=$UnetLr",
    "--text_encoder_lr=$TextEncoderLr",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=0",
    "--optimizer_type=AdamW8bit",
    "--network_module=lycoris.kohya",
    "--network_dim=$NetworkDim",
    "--network_alpha=$NetworkAlpha",
    "--network_args",
    "conv_dim=$ConvDim",
    "conv_alpha=$ConvAlpha",
    "algo=locon",
    "dropout=0",
    "--cache_latents",
    "--gradient_checkpointing",
    "--max_data_loader_n_workers=2",
    "--persistent_data_loader_workers",
    "--seed=$Seed"
)

if ($Sdxl) {
    $commonArgs += @("--network_train_unet_only", "--cache_text_encoder_outputs")
}

Write-Host "Image/caption pairs: $($images.Count)"
Write-Host "Dataset config: $datasetConfigPath"
Write-Host "Output directory: $OutputDir"
Write-Host ""
Write-Host "Command:"
Write-Host ("accelerate " + (($commonArgs | ForEach-Object {
    if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
}) -join " "))

if (-not $DryRun) {
    Push-Location $sdScriptsPath
    try {
        & accelerate @commonArgs
    }
    finally {
        Pop-Location
    }
}
