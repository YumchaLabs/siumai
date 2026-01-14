# Releasing (release-plz)

This repository uses [`release-plz`](https://github.com/release-plz/release-plz) to manage releases for a multi-crate Cargo workspace.

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

## How to publish a release

1. Merge the desired changes into `main`.
2. Go to **Actions** → **Release-plz** → **Run workflow**.
3. Set:
   - `release = true`
   - `dry_run = false`

This runs `release-plz release` to publish crates to crates.io and create the `siumai` tag + GitHub Release.

## About `dry_run`

`dry_run = true` maps to `cargo publish --dry-run`. It does **not** upload any crates.

If the workspace contains crates that are not yet present on crates.io (e.g. new crates introduced by a workspace split),
`--dry-run` can fail because Cargo still needs to resolve workspace dependencies from crates.io.

In that bootstrap scenario, run the release with `dry_run = false` to publish the dependency crates first.

Note: the workflow passes the `dry_run` input to the action only when it is explicitly set to `true`,
to avoid accidentally enabling dry-run due to string input handling.

## Why there may be no release PR

`release-plz release-pr` opens a PR when it needs to bump versions and/or update changelogs.

If versions were already bumped on `main` (e.g. during a migration), `release-pr` can be a no-op and no PR will be created.
