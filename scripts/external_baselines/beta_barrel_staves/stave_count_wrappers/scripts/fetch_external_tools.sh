#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/tools"
mkdir -p "$TOOLS"

verify_sha256() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing artifact for checksum verification: $path" >&2
    return 1
  fi
  echo "$expected  $path" | sha256sum -c -
}

checkout_repo() {
  local url="$1"
  local dest="$2"
  local ref="$3"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" fetch --depth 1 origin "$ref"
  else
    git clone --no-checkout "$url" "$dest"
    git -C "$dest" fetch --depth 1 origin "$ref"
  fi
  git -C "$dest" checkout --force FETCH_HEAD
  git -C "$dest" rev-parse HEAD
}

checkout_repo https://github.com/SluskyLab/PolarBearal3.git "$TOOLS/PolarBearal3" e8476c3d8bed6c07bb0f0b7e604be183eafe3a15
checkout_repo https://github.com/BernhoferM/TMbed.git "$TOOLS/TMbed" 8cee893523eb655bc9485c00c65336d27a236191
checkout_repo https://github.com/BolognaBiocomp/betaware.git "$TOOLS/betaware" c2797d595798c2febd30bacbde73222a01c4119b
checkout_repo https://github.com/pbagos/juchmme.git "$TOOLS/juchmme" 07d61d4c7627233b54b0206aad25a65d661d7fc7

JUCHMME_REL="$TOOLS/juchmme_release"
JUCHMME_ZIP="$JUCHMME_REL/juchmme_v1.0.6.zip"
JUCHMME_ZIP_SHA256="241266fe0341588be564924cce2bebddfa383152c64ac0138d4539d751a3ea2a"
mkdir -p "$JUCHMME_REL"
if [[ ! -d "$JUCHMME_REL/juchmme_git" ]]; then
  curl -L --fail \
    https://github.com/pbagos/juchmme/releases/download/1.0.6/juchmme_v1.0.6.zip \
    -o "$JUCHMME_ZIP"
  verify_sha256 "$JUCHMME_ZIP" "$JUCHMME_ZIP_SHA256"
  python -m zipfile -e "$JUCHMME_ZIP" "$JUCHMME_REL"
else
  verify_sha256 "$JUCHMME_ZIP" "$JUCHMME_ZIP_SHA256"
fi

PROFTMB="$TOOLS/proftmb_deb"
mkdir -p "$PROFTMB"
if [[ ! -x "$PROFTMB/root/usr/bin/proftmb" ]]; then
  (
    cd "$PROFTMB"
    apt-get download proftmb
    for pkg in libgsl27 libgslcblas0 libgsl25; do
      apt-get download "$pkg" || true
    done
    mkdir -p root
    for deb in ./*.deb; do
      dpkg-deb -x "$deb" root
    done
  )
fi
verify_sha256 "$PROFTMB/proftmb_1.1.12-11_amd64.deb" d284628626dd33cbd56eb3c0f33651641a322b0626dde526f81920c4e8279cec
verify_sha256 "$PROFTMB/libgsl27_2.7.1+dfsg-6ubuntu2_amd64.deb" a633774cf0eb1e5cd9fa7671da18ce503ef4d31585d52e27c56de7effcdb321e
verify_sha256 "$PROFTMB/libgslcblas0_2.7.1+dfsg-6ubuntu2_amd64.deb" 34ba0c31dd59224734a8cfba704ac381a11bcdb2ba91ff09888e8eda173617d4

echo "Fetched external tools into $TOOLS"
