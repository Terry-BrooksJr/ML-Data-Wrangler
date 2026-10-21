#!/usr/bin/env bash
# Wraps dist/Lasso.app (from packaging/build.sh) in a macOS installer package.
#
#   packaging/build_pkg.sh
#
# Step 1, pkgbuild: a "component package" holding the app and where it goes
# (/Applications), in build/pkg/. Step 2, productbuild: the installer people
# double-click, with its title and pages, as dist/Lasso-<version>.pkg.
set -euo pipefail
cd "$(dirname "$0")/.."

APP="dist/Lasso.app"
OUT="build/pkg"
if [[ ! -d "$APP" ]]; then
    echo "error: $APP not found. Build the app first: packaging/build.sh" >&2
    exit 1
fi

# The installer reuses the app's own ID and version, read from its Info.plist.
PLIST="$APP/Contents/Info.plist"
IDENTIFIER="$(/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" "$PLIST")"
VERSION="$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$PLIST")"

rm -rf "$OUT"
mkdir -p "$OUT"

# pkgbuild installs everything in its --root folder, and dist/ also holds
# PyInstaller's intermediate dist/Lasso/ folder. Stage a root with only the
# app. ditto copies the way macOS expects (permissions, code signature,
# symlinks inside the bundle).
ROOT="$OUT/root"
mkdir -p "$ROOT"
ditto "$APP" "$ROOT/Lasso.app"

# pkgbuild marks an app "relocatable" by default: if a copy of Lasso.app
# already exists anywhere on the Mac (say, dist/Lasso.app in this repo), the
# installer updates that copy instead of installing to /Applications. Write
# out pkgbuild's settings for the app and turn that off.
COMPONENT_PLIST="$OUT/component.plist"
pkgbuild --analyze --root "$ROOT" "$COMPONENT_PLIST" >/dev/null
/usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable false" "$COMPONENT_PLIST"

# --root: the folder whose contents are installed. --install-location: where
# they go, so Lasso.app lands at /Applications/Lasso.app.
pkgbuild \
    --root "$ROOT" \
    --component-plist "$COMPONENT_PLIST" \
    --identifier "$IDENTIFIER" \
    --version "$VERSION" \
    --install-location /Applications \
    "$OUT/Lasso-component.pkg"

# productbuild wraps the component package in the installer people open:
# packaging/installer/distribution.xml describes it (title, pages, the Apple
# Silicon and macOS 12.3 checks); --resources is the folder its welcome and
# conclusion pages are in; --package-path is where it finds Lasso-component.pkg.
sed -e "s/@IDENTIFIER@/$IDENTIFIER/g" -e "s/@VERSION@/$VERSION/g" \
    packaging/installer/distribution.xml >"$OUT/distribution.xml"
PKG="dist/Lasso-$VERSION.pkg"
productbuild \
    --distribution "$OUT/distribution.xml" \
    --resources packaging/installer/resources \
    --package-path "$OUT" \
    "$PKG"

echo "Built: $PKG"
