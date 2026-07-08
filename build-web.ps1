# build-web.ps1
# Build the WebAssembly module with correct paths and cleanup redundant directories.

Write-Host "Building WebAssembly module..." -ForegroundColor Cyan
# Run wasm-pack build targeting web and outputting to the root docs/pkg directory
wasm-pack build (Join-Path $PSScriptRoot "web") --target web --out-dir (Join-Path $PSScriptRoot "docs/pkg")

Write-Host "Cleaning up accidental/local duplicate outputs..." -ForegroundColor Cyan
$duplicatePkg = Join-Path $PSScriptRoot "web/pkg"
$duplicateDocs = Join-Path $PSScriptRoot "web/docs"
$redundantPk = Join-Path $PSScriptRoot "docs/pk"

if (Test-Path $duplicatePkg) {
    Remove-Item $duplicatePkg -Recurse -Force
    Write-Host "Removed duplicate output at $duplicatePkg" -ForegroundColor Yellow
}
if (Test-Path $duplicateDocs) {
    Remove-Item $duplicateDocs -Recurse -Force
    Write-Host "Removed duplicate output at $duplicateDocs" -ForegroundColor Yellow
}
if (Test-Path $redundantPk) {
    Remove-Item $redundantPk -Recurse -Force
    Write-Host "Removed redundant output at $redundantPk" -ForegroundColor Yellow
}

Write-Host "Success! WASM build output is ready in docs/pkg/." -ForegroundColor Green
