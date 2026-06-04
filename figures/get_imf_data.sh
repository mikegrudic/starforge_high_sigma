#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/imf_data"

while IFS= read -r src_dir || [[ -n "$src_dir" ]]; do
    [[ -z "$src_dir" ]] && continue
    rel="${src_dir#*STARFORGE_RT/}"
    rel="STARFORGE_RT/${rel}"
    dest_dir="${DEST}/${rel}"
    mkdir -p "${dest_dir}"
    cp "${src_dir}/IMF.dat" "${dest_dir}/IMF.dat"
    cp "${src_dir}/global_statistics.fits" "${dest_dir}/global_statistics.fits"
done < "${SCRIPT_DIR}/simulation_paths"
