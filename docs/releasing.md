# Releasing (release-plz)

This repository uses [`release-plz`](https://github.com/release-plz/release-plz) to manage releases for a multi-crate Cargo workspace.

## Release preflight

Run the maintained release gates serially from a clean release candidate:

```text
cargo fmt --all -- --check
python3 -B -m unittest discover -s scripts/tests -p "test_*.py"
python3 -B scripts/test-workspace.py flagship --runner nextest
python3 -B scripts/test-workspace.py full --runner nextest
cargo clippy --workspace --all-targets --all-features -j 1 -- -D warnings
cargo check -p siumai --no-default-features --lib -j 1
cargo check -p siumai --no-default-features --features all-providers --lib -j 1
cargo check -p siumai --no-default-features --features openai-responses-websocket,openai-realtime --lib -j 1
cargo check -p siumai --no-default-features --features openai --example openai_flagship -j 1
cargo check -p siumai --no-default-features --features anthropic --example anthropic_flagship -j 1
cargo check -p siumai --no-default-features --features openai,anthropic,registry --example provider_switching -j 1
cargo doc -p siumai-provider-openai --all-features --no-deps -j 1
cargo doc --workspace --all-features --no-deps -j 1
cargo test --doc --workspace --all-features -j 1
python3 -B scripts/check_package_file_list.py
```

Also run the CI MSRV lane using the `rust-version` declared by the workspace and inspect
`cargo metadata --locked --no-deps` after any dependency or feature change. Cargo and release-plz
are the authorities for package versions; repository scripts must not maintain a second release
version. These checks are deterministic and offline; credentialed provider tests are not a release
prerequisite unless a maintainer explicitly authorizes the external calls.

Pull requests run the fast suite followed by the exact OpenAI/Anthropic flagship package suite.
They also compile the facade without default features for bare, OpenAI, Anthropic, all-provider,
and combined Responses WebSocket/Realtime feature ownership paths. The OpenAI and Anthropic
flagship examples are each compiled with only their exact provider feature. The
`provider_switching` example is compiled with exactly OpenAI, Anthropic, and Registry to protect the
canonical concrete-to-erased facade journey. This remains a small fixed Cargo-native gate rather
than a provider-by-feature matrix or a custom public-symbol policy script. Pull requests also run
the documentation lane, which builds the OpenAI provider with all optional modules enabled before
workspace docs and doctests pass. Deterministic nextest failures are not retried, and the workspace
has a finite global test timeout.

The `flagship` lane validates a bounded OpenAI and Anthropic package slice. Passing it means
`claimed slice complete` for those deterministic gates; it is not a `provider platform complete`
claim. Product surfaces outside the documented package slice remain `intentionally deferred`.

Before publishing, validate the Cargo-native workspace file list with
`python3 -B scripts/check_package_file_list.py`. The checker invokes Cargo itself, verifies its
exit status, and rejects a small set of credential, private local configuration, repository-local,
editor-state, and live-canary artifact paths. It does not read package contents, decide membership,
or infer publication order; Cargo and release-plz remain the authorities. For a dirty local release
candidate, add `--allow-dirty` to the checker command; this changes only Cargo's local cleanliness
check and never authorizes publishing.

Run `cargo package --workspace --locked --allow-dirty -j 1` as the local package dry run where
crates.io dependency resolution permits it. New unpublished workspace dependencies can make this
command fail even when package contents and the workspace graph are correct; record that bootstrap
limitation separately and use the manual release-plz dry-run to exercise the maintained
dependency-ordered release path. Neither command publishes, tags, pushes, or opens a release PR.

Every package must carry the workspace license, repository, edition, MSRV, and a useful crate
README/rustdoc entry point. The facade's documented feature set must match its `[package.metadata.docs.rs]`
configuration. A breaking release updates the root changelog and migration guide together. The first
published version after this API reset becomes the new semver baseline; do not hide intentional
breaks behind compatibility aliases merely to satisfy the previous beta baseline.

The compatibility boundary established by
[ADR 0019](adr/0019-facade-family-call-ownership.md) includes the six root family modules, their
default operations, their bound `call` entry points and public call types, complete family response
types, and typed provider-option ownership. A later beta release that breaks this boundary must not
enter the standard release flow until review confirms all of the following:

- an architecture rationale names every affected canonical symbol;
- the migration guide contains an exact old-symbol-to-new-symbol map and behavioral notes;
- the root README, facade rustdoc, and compile-checked examples teach the replacement path;
- the changelog identifies the break, and Cargo-native no-default/provider/Registry gates cover it.

Compatibility aliases are not required, but migration evidence is. Release review should stop on a
canonical rename or ownership move that lacks any item above.

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
The workflow keeps publication order, tag creation, and GitHub Release creation inside that one
pinned native command rather than maintaining a second publication or tag engine.

The pinned `release-plz` 0.3.157 release command is not a general repair tool for partially created
repository releases. It treats any local tag with the expected name as already handled before it
checks the tag target or GitHub Release state. If a run leaves an exact tag without a GitHub Release,
or a same-name tag points at the wrong commit, stop and inspect the repository state instead of
rerunning or wrapping the command. Do not pre-create, move, or delete release tags as an automated
workaround.

The manual job therefore runs only from `refs/heads/main`, verifies that the checkout still equals
the current remote `main` commit, and rejects any pre-existing tag for the workspace version before
installing release tooling or using publish credentials. Partially published crates can converge on
a later run while that tag is absent. A tag-only state is intentionally a visible manual stop, not
an automated recovery path.

The release PR is the preferred way to prepare version and changelog changes, but it is not a
publishing authorization boundary. The actual publish job is manually dispatched from `main`, and
`release_always = true` allows it to publish an unpublished workspace version even when a workflow
fix or another normal PR landed after the release PR.

## Standard release flow

1. Merge the desired changes into `main`.
2. Wait for **Release-plz PR** to create or update the release PR when version or changelog changes
   are still needed.
3. If a release PR is created, review it:
   - version bumps
   - root `CHANGELOG.md`
   - crate changelogs
   - migration notes, when the public API changed
   - CI results
4. Merge the release PR when one exists. If no release PR is created, verify that `main` already
   contains the intended workspace version and changelog entries.
5. Go to **Actions** → **Release-plz** → **Run workflow** on `main`.
6. Set:
   - `release = true`
   - `dry_run = false`
7. Verify:
   - all expected crates are published on crates.io
   - the `siumai` tag exists in the repository
   - the GitHub Release exists and uses the expected changelog section

This runs the preflight above and then one pinned `release-plz release` command to publish crates to
crates.io and create the `siumai` tag + GitHub Release.

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

The release workflow retries automatically on 429 by waiting until the timestamp suggested by
crates.io and then re-running the same `release-plz release`. Only a bounded diagnostic suffix is
retained for classification; non-429 failures stop immediately.

## Why there may be no release PR

`release-plz release-pr` opens a PR when it needs to bump versions and/or update changelogs.

If versions were already bumped on `main` (for example during a migration), `release-pr` can be a
no-op and no PR will be created. In that case, do not create a tag manually. Verify the version and
changelog state on `main`, run the manual dry-run when it is meaningful for the dependency graph,
then use the same manual release workflow. `release-plz release` remains responsible for publishing
unpublished crates and creating the repository tag and GitHub Release.
