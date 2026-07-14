#!/usr/bin/env bash
# download-plaicraft.sh — Download PLAICraft public dataset (metadata + small slice)
# Run on AMD Developer Cloud (MI300X) where disk space is available.
#
# Usage:
#   ./download-plaicraft.sh metadata     # Download only metadata SQLite (~MB)
#   ./download-plaicraft.sh slice <N>    # Download N sessions of raw data (~GB each)
#   ./download-plaicraft.sh full         # Download full 621GB zip (WARNING: slow)
set -euo pipefail

BUCKET="plai-public-data-bucket-prod"
REGION="s3.us-west-2.amazonaws.com"
DEST="${PLAICRAFT_DIR:-./plaicraft-data}"
ZIP_URL="https://${BUCKET}.${REGION}/plaicraft_dante_morgan_xander_200hrs.zip"

mkdir -p "${DEST}"

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found. Install: pip install awscli"
  echo "  Then: aws configure (or use: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env)"
  exit 1
fi

list_bucket() {
  echo "[INFO] Listing S3 bucket contents (first 50 keys)..."
  aws s3 ls "s3://${BUCKET}/" --no-sign-request --human-readable 2>/dev/null | head -50 || {
    echo "[WARN] Could not list bucket. Trying public HTTP listing..."
    curl -s "https://${BUCKET}.${REGION}/?list-type=2&max-keys=50" 2>/dev/null | head -100
  }
}

download_metadata() {
  echo "[INFO] Downloading PLAICraft metadata (SQLite databases)..."
  # Try listing individual metadata files from the bucket
  aws s3 ls "s3://${BUCKET}/" --no-sign-request --recursive 2>/dev/null | grep -i "metadata\|\.db\|\.sqlite" | head -20 || {
    echo "[WARN] Cannot list metadata files individually."
    echo "[INFO] The 621GB zip may be the only distribution method."
    echo "[INFO] Trying to download just the metadata portion via HTTP range..."
  }
  # Save listing for inspection
  aws s3 ls "s3://${BUCKET}/" --no-sign-request --recursive 2>/dev/null > "${DEST}/bucket-listing.txt" || true
  echo "[INFO] Bucket listing saved to ${DEST}/bucket-listing.txt"
}

download_slice() {
  local n="${1:-2}"
  echo "[INFO] Attempting to download ${n} sessions from PLAICraft..."
  echo "[INFO] If individual files are not accessible, use 'full' mode."
  list_bucket

  # Try downloading individual session directories
  local downloaded=0
  while IFS= read -r line; do
    local key
    key=$(echo "${line}" | awk '{print $NF}')
    if [[ "${key}" == *"/" ]] || [[ -z "${key}" ]]; then continue; fi
    if [[ "${key}" == *.db ]] || [[ "${key}" == *.sqlite ]]; then
      echo "[INFO] Downloading metadata: ${key}"
      aws s3 cp "s3://${BUCKET}/${key}" "${DEST}/${key}" --no-sign-request --quiet || true
    fi
  done < "${DEST}/bucket-listing.txt"

  echo "[INFO] Metadata download complete. For raw video/audio, use 'full' mode."
}

download_full() {
  echo "[WARN] Downloading full PLAICraft zip (621GB). This will take a LONG time."
  echo "[WARN] Ensure you have >700GB free disk space."
  local free_gb
  free_gb=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')
  if [[ "${free_gb}" -lt 700 ]]; then
    echo "[ERROR] Only ${free_gb}GB free. Need 700GB+."
    exit 1
  fi
  echo "[INFO] Starting download to ${DEST}/plaicraft_200hrs.zip"
  curl -L -C - -o "${DEST}/plaicraft_200hrs.zip" "${ZIP_URL}"
  echo "[INFO] Download complete. Extract with: unzip -l to inspect, then selective unzip."
}

case "${1:-help}" in
  metadata)
    download_metadata
    ;;
  slice)
    download_slice "${2:-2}"
    ;;
  full)
    download_full
    ;;
  list)
    list_bucket
    ;;
  help|*)
    cat <<EOF
PLAICraft Downloader

Usage:
  $0 metadata        Download metadata SQLite only (~MB)
  $0 slice [N]       Download N sessions of raw data (~GB)
  $0 full            Download full 621GB zip (slow!)
  $0 list            List bucket contents

Environment:
  PLAICRAFT_DIR      Destination directory (default: ./plaicraft-data)
EOF
    ;;
esac
