#!/bin/bash
###############################################################################
#
# A bash script to extract only the notes related to the most recent version of
# earthdata-hashdiff from CHANGELOG.md
#
# 2023-06-16: Created.
# 2023-10-10: Copied from earthdata-varinfo repository to HOSS.
# 2024-01-03: Copied from HOSS repository to the Swath Projector.
# 2024-01-23: Copied and modified from Swath Projector repository to HyBIG.
# 2025-07-15: Copied and modified from HyBIG to earthdata-hashdiff.
# 2025-09-05: Copied and modified from earthdata-hashdiff to HGA.
#
###############################################################################

CHANGELOG_FILE="CHANGELOG.md"

## captures versions
## >## v1.0.0
## >## [v1.0.0]
VERSION_PATTERN="^## [\[]v"

## captures url links
## [v1.2.0]: https://github.com/nasa/harmony-gdal-adapter/releases/tags/1.2.0
LINK_PATTERN="^\[.*\]:.*https://github.com/nasa"

# Read the file and extract text between the first two occurrences of the
# VERSION_PATTERN
result=$(awk "/$VERSION_PATTERN/{c++; if(c==2) exit;} c==1" "$CHANGELOG_FILE")

# Print the result
echo "$result" |  grep -v "$VERSION_PATTERN" | grep -v "$LINK_PATTERN"
