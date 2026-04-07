#!/bin/bash
# Push a Flash-MoE model directory to iPhone via USB (with resume)
# Usage: ./push_model_to_iphone.sh /path/to/model [bundle_id]
# Re-run to resume — skips files already on device.

set -e

SRC="${1:-/Volumes/BK/flash-moe-397B}"
BUNDLE="${2:-com.alexintosh.flashmoe}"

ts() { date "+%H:%M:%S"; }

if [ ! -d "$SRC" ]; then
    echo "ERROR: Model directory not found: $SRC"
    exit 1
fi

MODEL_NAME=$(basename "$SRC")
TOTAL_FILES=$(find "$SRC" -type f | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$SRC" | cut -f1)

echo "=== Flash-MoE iPhone Model Push ==="
echo "Source:    $SRC"
echo "Model:    $MODEL_NAME"
echo "Bundle:   $BUNDLE"
echo "Files:    $TOTAL_FILES"
echo "Size:     $TOTAL_SIZE"
echo ""

# Check device is connected
if ! pymobiledevice3 usbmux list 2>/dev/null | grep -q "DeviceClass"; then
    echo "ERROR: No iPhone connected. Plug in via USB and trust this computer."
    exit 1
fi

echo "$(ts) iPhone detected."

# Check for --force flag
FORCE=0
if [ "$3" = "--force" ] || [ "$2" = "--force" ]; then
    FORCE=1
    echo "$(ts) Force mode — will overwrite existing files"
fi

# Get list of files already on device (for resume)
echo "$(ts) Checking existing files on device..."
EXISTING=$(python3 -c "
import os
from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.house_arrest import HouseArrestService
lockdown = create_using_usbmux()
afc = HouseArrestService(lockdown=lockdown, bundle_id='$BUNDLE')
# Create all needed subdirectories
for subdir in ['packed_experts', 'packed_experts_tiered', 'packed_experts_2bit']:
    try: afc.makedirs('Documents/$MODEL_NAME/' + subdir)
    except: pass
# List all files recursively
def list_recursive(path, prefix=''):
    try:
        for f in afc.listdir(path):
            if f.startswith('.'): continue
            full = path + '/' + f
            rel = prefix + f if not prefix else prefix + '/' + f
            try:
                afc.listdir(full)  # if this works, it's a directory
                list_recursive(full, rel + '/')
            except:
                print(rel if prefix else f)
    except: pass
list_recursive('Documents/$MODEL_NAME')
" 2>/dev/null)

if [ "$FORCE" = "1" ]; then
    EXISTING=""
fi

EXISTING_COUNT=$(echo "$EXISTING" | grep -c . || echo 0)
echo "$(ts) $EXISTING_COUNT files already on device"

# Push files one by one, skipping existing
COUNT=0
SKIPPED=0
FAILED=0
find "$SRC" -type f | sort | while read -r FILE; do
    REL_PATH="${FILE#$SRC/}"
    REMOTE_PATH="Documents/${MODEL_NAME}/${REL_PATH}"
    COUNT=$((COUNT + 1))

    # Skip if already on device
    if echo "$EXISTING" | grep -qx "$REL_PATH"; then
        echo "$(ts) [$COUNT/$TOTAL_FILES] $REL_PATH — SKIP (exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    SIZE=$(du -h "$FILE" | cut -f1)
    echo -n "$(ts) [$COUNT/$TOTAL_FILES] $REL_PATH ($SIZE) ... "

    if pymobiledevice3 apps push "$BUNDLE" "$FILE" "$REMOTE_PATH" 2>/dev/null; then
        echo "OK"
    else
        echo "FAILED — retrying in 3s..."
        sleep 3
        if pymobiledevice3 apps push "$BUNDLE" "$FILE" "$REMOTE_PATH" 2>/dev/null; then
            echo "$(ts)   retry OK"
        else
            echo "$(ts)   retry FAILED — run script again to resume"
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo ""
echo "$(ts) === Transfer complete ==="
echo "Re-run to resume if any files failed."
