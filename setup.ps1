# יצירת תיקיות
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\models" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "src" | Out-Null

# יצירת קבצי קוד ריקים
$files = @(
    "src\config.py",
    "src\data.py",
    "src\features.py",
    "src\models.py",
    "src\train.py",
    "src\evaluate.py",
    "src\visualize.py",
    "src\utils.py",
    "src\main.py",
    "requirements.txt",
    "README.md"
)

foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
    }
}

Write-Host "✅ Project structure created successfully!"
