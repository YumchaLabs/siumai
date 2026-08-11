# Releasing (release-plz)

This repository uses [`release-plz`](https://github.com/release-plz/release-plz) to manage releases for a multi-crate Cargo workspace.

## Release preflight

Run the maintained release gates serially from a clean release candidate:

```text
cargo fmt --all -- --check
python3 -B -m unittest discover -s scripts/tests -p "test_*.py"
python3 -B scripts/check_workspace_boundaries.py --target
python3 -B scripts/test-workspace.py flagship --runner nextest
python3 -B scripts/test-workspace.py full --runner nextest
cargo clippy --workspace --all-targets --all-features -j 1 -- -D warnings
cargo doc --workspace --all-features --no-deps -j 1
cargo test --doc --workspace --all-features -j 1
```

Also run the CI MSRV lane with Rust 1.88 and inspect `cargo metadata --locked --no-deps` after any
dependency or feature change. These checks are deterministic and offline; credentialed provider
tests are not a release prerequisite unless a maintainer explicitly authorizes the external calls.

Pull requests run the fast suite followed by the exact OpenAI/Anthropic flagship package suite.
They also compile the facade without default features for bare, OpenAI, Anthropic, and the combined
Responses WebSocket/Realtime feature ownership paths. This remains a small fixed gate rather than a
provider-by-feature matrix.

Before publishing, verify package contents with `cargo package --list -p <crate>` for changed crates
and run `cargo package -p <crate> --allow-dirty` in dependency order where crates.io dependency
resolution permits it. New unpublished workspace dependencies can make local dry runs fail even when
the package graph is correct; the manual release job handles the real dependency-ordered publish.

Every package must carry the workspace license, repository, edition, MSRV, and a useful crate
README/rustdoc entry point. The facade's documented feature set must match its `[package.metadata.docs.rs]`
configuration. A breaking release updates the root changelog and migration guide together. The first
published version after this API reset becomes the new semver baseline; do not hide intentional
breaks behind compatibility aliases merely to satisfy the previous beta baseline.

## What gets released

- **Crates.io**: all unpublished workspace crates are published in dependency order.
- **Git tag + GitHub Release**: only the facade crate `siumai` creates a repository tag and a GitHub Release (tag format: `v{{ version }}`).
  - Other crates are still published to crates.io, but they don't create repo tags/releases.

## Required secrets

### Publishing to crates.io

- `CARGO_REGISTRY_TOKEN`: a crates.io API token with publish permissions for all crates in this workspace.

### Creating PRs / tags / GitHub Releases

Release-plz needs a GitHub token with write access.

Recommended (GitHub App, best for orgs):
- `RELEASE_PLZ_APP_ID`
- `RELEASE_PLZ_APP_PRIVATE_KEY`

Alternative (PAT):
- `RELEASE_PLZ_TOKEN`

Fallback (not recommended):
- `GITHUB_TOKEN` (may not trigger downstream workflows for PRs/tags depending on repo settings).

## Release policy

Do **not** create or push release tags manually.

In this repository, release tags are an output of `release-plz release`, not the trigger for publishing.
This keeps crates.io publishing, the `v{{ version }}` git tag, and the GitHub Release synchronized.

The long-term configuration uses `release_always = false`, so releases should go through the release-plz
PR flow before the manual publish job runs.

## Standard release flow

1. Merge the desired changes into `main`.
2. Wait for **Release-plz PR** to create or update the release PR.
3. Review the release PR:
   - version bumps
   - root `CHANGELOG.md`
   - crate changelogs
   - migration notes, when the public API changed
   - CI results
4. Merge the release PR.
5. Go to **Actions** → **Release-plz** → **Run workflow** on `main`.
6. Set:
   - `release = true`
   - `dry_run = false`
7. Verify:
   - all expected crates are published on crates.io
   - the `siumai` tag exists in the repository
   - the GitHub Release exists and uses the expected changelog section

This runs `release-plz release` to publish crates to crates.io and create the `siumai` tag + GitHub Release.

## Manual dry run

Use **Actions** → **Release-plz** → **Run workflow** with:

- `release = true`
- `dry_run = true`

This maps to `cargo publish --dry-run`. It does **not** upload any crates or create a release tag.

If the workspace contains crates that are not yet present on crates.io (e.g. new crates introduced by a workspace split),
`--dry-run` can fail because Cargo still needs to resolve workspace dependencies from crates.io.

In that bootstrap scenario, run the release with `dry_run = false` to publish the dependency crates first.

Note: the workflow passes the `dry_run` input to the action only when it is explicitly set to `true`,
to avoid accidentally enabling dry-run due to string input handling.

## Crates.io 429 (rate limit)

When publishing many new crates (common during a workspace split), crates.io can return `429 Too Many Requests`.

The release workflow retries automatically on 429 by waiting until the timestamp suggested by crates.io and then re-running `release-plz release`.

## Why there may be no release PR

`release-plz release-pr` opens a PR when it needs to bump versions and/or update changelogs.

If versions were already bumped on `main` (e.g. during a migration), `release-pr` can be a no-op and no PR will be created.
In that case, do not create a tag manually; inspect the Release-plz logs and decide whether the version/changelog
state should be corrected with a normal PR.
