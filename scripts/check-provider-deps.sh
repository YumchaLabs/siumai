#!/usr/bin/env zsh
set -euo pipefail

# Guardrail: provider crates should not depend on other provider crates.
#
# Allowed exceptions:
# - `siumai-provider-openai-compatible` is a shared "family" crate for OpenAI-like protocol mapping.

allowed_deps=(
  "siumai-provider-openai-compatible"
)

is_allowed_dep() {
  local dep="$1"
  for allowed in "${allowed_deps[@]}"; do
    if [[ "${dep}" == "${allowed}" ]]; then
      return 0
    fi
  done
  return 1
}

extract_provider_deps_from_section() {
  local toml="$1"
  local section="$2"
  # Print dependency keys (left-hand side) matching `siumai-provider-* = ...` within the section.
  awk -v section="${section}" '
    $0 ~ "^\\[" section "\\]$" { inside=1; next }
    $0 ~ "^\\[" && inside==1 { exit }
    inside==1 { print }
  ' "${toml}" | \
    sed -nE 's/^[[:space:]]*(siumai-provider-[a-z0-9_-]+)[[:space:]]*=.*$/\1/p' | \
    sort -u || true
}

fail=0

for toml in siumai-provider-*/Cargo.toml; do
  [[ -f "${toml}" ]] || continue

  crate_name="$(
    rg -m 1 '^name = "siumai-provider-' "${toml}" 2>/dev/null \
      | sed -E 's/^name = "([^"]+)".*$/\\1/'
  )"
  [[ -n "${crate_name}" ]] || continue

  deps=()
  deps+=($(extract_provider_deps_from_section "${toml}" "dependencies" || true))
  deps+=($(extract_provider_deps_from_section "${toml}" "dev-dependencies" || true))
  deps+=($(extract_provider_deps_from_section "${toml}" "build-dependencies" || true))

  for dep in "${deps[@]}"; do
    if [[ "${dep}" == "${crate_name}" ]]; then
      continue
    fi
    if is_allowed_dep "${dep}"; then
      continue
    fi
    echo "[check-provider-deps] ERROR: ${crate_name} depends on ${dep} (provider→provider dependency is disallowed)."
    echo "[check-provider-deps]   file: ${toml}"
    fail=1
  done
done

if [[ "${fail}" != "0" ]]; then
  echo "[check-provider-deps] FAILED"
  exit 1
fi

echo "[check-provider-deps] OK"
