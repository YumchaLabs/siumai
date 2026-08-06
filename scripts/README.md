# Repository Scripts

All maintained local-development entry points are Python 3 scripts so the same command works on
Windows, macOS, and Linux. Rust test commands run serially with `-j 1`; when available,
`cargo-nextest` is preferred automatically.

The examples below use `python3`; on Windows, use `py -3` if Python is installed through the
standard Python launcher.

## Local test suites

`test-workspace.py` is the main local entry point:

```text
# Fast core, transport, registry, and facade loop
python3 scripts/test-workspace.py fast

# One release-level workspace/all-features suite
python3 scripts/test-workspace.py full
```

Useful options:

- `--runner auto|nextest|cargo-test` selects the Rust test runner.
- `--dry-run` prints the exact commands without executing them.

Provider work should use the owning package directly, for example:

```text
cargo nextest run -p siumai-provider-minimax --all-features -j 1
```

Credentialed or live-provider tests remain explicit provider-owned targets and are never part of
the default local suite.

## Architecture checks

```text
python3 -B scripts/check_workspace_boundaries.py --target
python3 -B -m unittest discover -s scripts/tests -p "test_*.py"
```

These checks validate declared package boundaries and the small Python entry points. Protocol
fixtures live beside their owning crate and are exercised directly by Rust tests; there is no
separate inventory of unused snapshots. The scripts intentionally do not parse Rust source or
attempt to infer compiler semantics.

## Release retry

`release_plz_release_with_retry.py` is the release-only retry wrapper used by GitHub Actions. It
streams `release-plz` output, recognizes crates.io rate limiting, parses the retry timestamp with
the Python standard library, and applies bounded retries without depending on Bash or GNU `date`.
The workflow also uses `--dry-run` on this same Python entry point, so release automation has no
platform-specific shell wrapper.
