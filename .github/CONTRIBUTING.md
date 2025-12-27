# Contributing to Siumai

Thank you for your interest in contributing! 🎉

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/siumai.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. **Run code quality checks**:
   - `cargo fmt` - Format code
   - `cargo clippy` - Check for code issues
   - `cargo test` - Run all tests
6. Submit a pull request

## How to Contribute

- 🐛 **Bug Reports**: Use the Bug Report template
- ✨ **Feature Requests**: Use the Feature Request template
- 🤖 **New Providers**: Use the Provider Request template
- 📚 **Documentation**: Use the Documentation template

## Development Guidelines

- Follow standard Rust conventions
- **Run `cargo fmt` before marking PR as ready** - Ensure consistent formatting
- **Run `cargo clippy` before marking PR as ready** - Fix all warnings and errors
- **Run tests before marking PR as ready** - Ensure all tests pass
- Add tests for new features
- Update documentation when needed
- Test with real API keys when possible

## Code Quality Requirements

Before marking your pull request as **ready for review**, you **must** run:

```bash
# Format code according to Rust standards
cargo fmt

# Check for code issues and style violations
cargo clippy

# Run tests to ensure nothing is broken
./scripts/test-full.sh
```

All clippy warnings must be resolved and all tests must pass before your PR can be merged.

💡 **Tip**: You can create a Draft PR while still working on your changes, and only run these checks when you're ready for review.

### Fast iteration during refactors

When you're iterating on a refactor, prefer the faster safety net:

```bash
./scripts/test-fast.sh
```

## Commit Messages

Use conventional commits:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes

## Getting Help

- 💬 [GitHub Discussions](https://github.com/YumchaLabs/siumai/discussions)
- 🐛 [Issues](https://github.com/YumchaLabs/siumai/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
