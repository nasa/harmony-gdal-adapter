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
# 2025-09-22: Append git commit messages to release notes.
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

# Get all commit messages since the last release (marked with a git tag). If
# there are no tags, get the full commit history of the repository.
if [[ $(git tag) ]]
then
	# There are git tags, so get the most recent one
	GIT_REF=$(git describe --tags --abbrev=0)
else
	# There are not git tags, so get the initial commit of the repository
	GIT_REF=$(git rev-list --max-parents=0 HEAD)
fi

# Retrieve the title line of all commit messages since $GIT_REF, filtering out
# those from the pre-commit-ci[bot] author.
GIT_COMMIT_MESSAGES=$(git log --oneline --format="%s" --perl-regexp --author='^(?!pre-commit-ci\[bot\]).*$' ${GIT_REF}..HEAD)

# Append git commit messages to the release notes:
if [[ ${GIT_COMMIT_MESSAGES} ]]
then
	result+="\n\n## Commits\n\n${GIT_COMMIT_MESSAGES}"
fi

# Print the result
echo -e "$result" |  grep -v "$VERSION_PATTERN" | grep -v "$LINK_PATTERN"
