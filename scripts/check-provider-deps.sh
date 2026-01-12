#!/usr/bin/env zsh
set -euo pipefail

# Guardrail: provider crates should not depend on other provider crates.
#
# Allowed exceptions:
# - Legacy protocol crate names (compatibility aliases): `siumai-provider-openai-compatible`, `siumai-provider-anthropic-compatible`.
#
# Preferred protocol crate names (`siumai-protocol-*`) are not matched by the guardrail regex and are
# therefore always allowed.

allowed_deps=(
  "siumai-provider-openai-compatible"
  "siumai-provider-anthropic-compatible"
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

find_first_line() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -m 1 "${pattern}" "${file}" 2>/dev/null || true
  else
    grep -m 1 -E "${pattern}" "${file}" 2>/dev/null || true
  fi
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
    find_first_line '^name = "siumai-provider-' "${toml}" \
      | sed -E 's/^name = "([^"]+)".*$/\1/'
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
