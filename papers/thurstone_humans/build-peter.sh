#!/bin/sh
# Compile Peter's editing copy and open the PDF.
#   ./build-peter.sh
cd "$(dirname "$0")" || exit 1
tectonic -X compile paper-peter.tex && open paper-peter.pdf
