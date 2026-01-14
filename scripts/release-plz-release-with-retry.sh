#!/usr/bin/env bash
set -euo pipefail

max_attempts="${RELEASE_PLZ_MAX_ATTEMPTS:-10}"
min_sleep_seconds="${RELEASE_PLZ_MIN_SLEEP_SECONDS:-60}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}" >&2
    exit 2
  fi
}

require_env "GITHUB_TOKEN"

extract_retry_after_epoch() {
  local file="$1"
  local ts

  # Example:
  # "... status 429 Too Many Requests): ... Please try again after Wed, 14 Jan 2026 14:43:25 GMT ..."
  ts="$(grep -oE 'after [A-Za-z]{3}, [0-9]{1,2} [A-Za-z]{3} [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT' "$file" | head -n1 || true)"
  ts="${ts#after }"
  if [[ -z "$ts" ]]; then
    return 1
  fi

  date -d "$ts" +%s 2>/dev/null
}

is_crates_io_429() {
  local file="$1"
  grep -qE 'status 429 Too Many Requests|published too many new crates' "$file"
}

attempt=1
while (( attempt <= max_attempts )); do
  echo "::group::release-plz release attempt ${attempt}/${max_attempts}"
  tmp_out="$(mktemp)"

  set +e
  release-plz release --git-token "${GITHUB_TOKEN}" 2>&1 | tee "$tmp_out"
  status="${PIPESTATUS[0]}"
  set -e

  echo "::endgroup::"

  if [[ "$status" -eq 0 ]]; then
    echo "release-plz release succeeded."
    rm -f "$tmp_out"
    exit 0
  fi

  if ! is_crates_io_429 "$tmp_out"; then
    echo "release-plz release failed (non-429)."
    rm -f "$tmp_out"
    exit "$status"
  fi

  retry_after_epoch="$(extract_retry_after_epoch "$tmp_out" || true)"
  rm -f "$tmp_out"

  if [[ -n "${retry_after_epoch:-}" ]]; then
    now_epoch="$(date -u +%s)"
    sleep_seconds="$(( retry_after_epoch - now_epoch + 10 ))"
    if (( sleep_seconds < min_sleep_seconds )); then
      sleep_seconds="$min_sleep_seconds"
    fi
    echo "Hit crates.io rate limit (429). Sleeping ${sleep_seconds}s until retry window."
  else
    sleep_seconds="$min_sleep_seconds"
    echo "Hit crates.io rate limit (429). Sleeping ${sleep_seconds}s (fallback)."
  fi

  sleep "$sleep_seconds"
  attempt="$(( attempt + 1 ))"
done

echo "release-plz release kept hitting crates.io 429 after ${max_attempts} attempts."
exit 1

