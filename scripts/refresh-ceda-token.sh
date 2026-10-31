#!/usr/bin/env bash
# Fetch a fresh CEDA archive access token and write it where meteopack looks.
#
# CEDA tokens last about three days and cannot be refreshed, so this is a thing
# you run again rather than a thing you set up once. That is why it exists as a
# script: the alternative is clicking through the website every few days.
#
#   scripts/refresh-ceda-token.sh              # writes ~/.ceda_token
#   CEDA_TOKEN_FILE=/somewhere scripts/refresh-ceda-token.sh
#
# The password is read from the terminal, passed to curl on stdin, and never
# written anywhere: not to a file, not to the environment, and not to argv
# (`curl -u user:pass` would put it in the process list, where every other user
# on a shared node can read it with `ps`).
set -euo pipefail

TOKEN_URL="https://services.ceda.ac.uk/api/token/create/"
TEST_URL="https://dap.ceda.ac.uk/badc/ARCHIVE_INFO/ACCESS_TEST/RESTRICTED/TOKEN_CHECK"
DEST="${CEDA_TOKEN_FILE:-$HOME/.ceda_token}"

if [ -t 0 ]; then
  read -r -p "CEDA username: " CEDA_USER
  read -r -s -p "CEDA password: " CEDA_PASS; echo
else
  : "${CEDA_USERNAME:?set CEDA_USERNAME and CEDA_PASSWORD, or run this on a terminal}"
  : "${CEDA_PASSWORD:?set CEDA_USERNAME and CEDA_PASSWORD, or run this on a terminal}"
  CEDA_USER="$CEDA_USERNAME"; CEDA_PASS="$CEDA_PASSWORD"
fi

response=$(printf 'user = "%s:%s"\n' "$CEDA_USER" "$CEDA_PASS" |
  curl --silent --show-error --fail-with-body --location \
       --request POST "$TOKEN_URL" --config -)
unset CEDA_PASS

token=$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
if [ -z "$token" ]; then
  echo "no access_token in the response from $TOKEN_URL" >&2
  exit 1
fi

umask 077
printf '%s\n' "$token" > "$DEST"
chmod 600 "$DEST"
echo "wrote a token to $DEST (expires in about 3 days)"

# A token that parses is not a token that works: CEDA answers 200 with a login
# page when it rejects one, so check against the archive rather than trusting it.
if curl --silent --location --header "Authorization: Bearer $token" "$TEST_URL" |
     grep -qi "you have correctly"; then
  echo "verified against $TEST_URL"
else
  echo "WARNING: the token did not pass $TEST_URL." >&2
  echo "Your CEDA account may not have access to the dataset you want;" >&2
  echo "apply for it at https://services.ceda.ac.uk/ and try again." >&2
fi
