# Local CI/CD Testing Guide

This guide explains how to test GitHub Actions workflows locally using `act`, allowing you to validate your CI/CD pipelines before pushing to GitHub.

## Overview

`act` is a command-line tool that runs GitHub Actions workflows locally using Docker containers. This enables you to:

- Test workflows before pushing to GitHub
- Debug workflow issues locally
- Validate workflow changes quickly
- Save CI/CD minutes and resources

## Prerequisites

### 1. Docker

`act` requires Docker to run workflows. Install Docker in WSL:

```bash
# Install Docker in WSL
sudo apt-get update
sudo apt-get install docker.io
sudo service docker start

# Or use Docker Desktop for Windows (works with WSL2)
# Download from: https://www.docker.com/products/docker-desktop
```

Verify Docker is running:

```bash
docker --version
docker ps
```

### 2. Install `act`

```bash
# Install act using the official installer
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or using snap (if available)
sudo snap install act

# Or using Homebrew (if installed)
brew install act
```

### 3. Add `act` to PATH

If `act` is installed to `./bin/act`, add it to your PATH:

```bash
# Add to ~/.bashrc (permanent)
echo 'export PATH="$HOME/Independent_Research/pbf-lbm-nosql-data-warehouse/XCT_Thermomagnetic_Analysis/bin:$PATH"' >> ~/.bashrc

# Or add to current session (temporary)
export PATH="$PWD/bin:$PATH"

# Verify installation
act --version
```

## Basic Usage

### List All Workflows

See all available workflows and their events:

```bash
act -l
```

This shows:
- Workflow names
- Job names
- Available events (workflow_dispatch, schedule, etc.)

### Dry-Run Mode

Test workflows without actually running them (recommended first step):

```bash
# Test CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n

# Test PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Test nightly workflow (simulate schedule)
act schedule -W .github/workflows/nightly.yml -n
```

The `-n` flag shows what would happen without executing.

### Run Workflows

Actually execute workflows (takes longer, uses Docker):

```bash
# Run CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main

# Run PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main

# Run nightly workflow
act schedule -W .github/workflows/nightly.yml
```

## Testing Specific Workflows

### CI Workflow

The CI workflow runs tests, linting, and code quality checks:

```bash
# Dry-run
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n

# Full run
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main

# Test specific job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint
```

**Inputs:**
- `branch`: Choose branch (main, develop, or all)

### PR Workflow

The PR workflow validates pull requests:

```bash
# Dry-run
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Full run
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main
```

**Inputs:**
- `pr_number`: Pull request number (required)
- `branch`: Target branch (main or develop)

### Nightly Workflow

The nightly workflow runs weekly tests and security checks:

```bash
# Simulate schedule trigger
act schedule -W .github/workflows/nightly.yml -n

# Or trigger manually
act workflow_dispatch -W .github/workflows/nightly.yml -n
```

### Release Workflow

The release workflow builds and tests releases:

```bash
# Quick test (dry-run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0 -n

# Or full run
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0

# Simulate release event (alternative)
act release -W .github/workflows/release.yml -n
```

**Inputs:**
- `version`: Version tag (e.g., v1.0.0) - **required**

**Note**: The documentation build job only runs on actual `release` events, not `workflow_dispatch`. When testing manually, only the build-and-test job will run.

## Advanced Usage

### Test Specific Jobs

Run only specific jobs from a workflow:

```bash
# Run only the lint job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint

# Run only the test job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j test

# Run multiple jobs
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint -j test
```

### Use Different Docker Images

Specify a different runner image:

```bash
# Use medium image (default, recommended)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -P ubuntu-latest=catthehacker/ubuntu:act-latest

# Use large image (more compatible, but ~17GB)
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -P ubuntu-latest=catthehacker/ubuntu:act-22.04
```

### Set Environment Variables

Pass environment variables to workflows:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -e .env
```

Or set inline:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main --env MY_VAR=value
```

### Verbose Output

Get more detailed output:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v
```

### List Available Events

See what events a workflow supports:

```bash
act -l -W .github/workflows/ci.yml
```

## Configuration

### Act Configuration File

Create `~/.config/act/actrc` to set defaults:

```bash
mkdir -p ~/.config/act
cat > ~/.config/act/actrc << EOF
-P ubuntu-latest=catthehacker/ubuntu:act-latest
--container-architecture linux/amd64
EOF
```

### Docker Image Selection

When first running `act`, you'll be prompted to choose a default image:

- **Large size image**: ~17GB download, most compatible, includes snapshots of GitHub Hosted Runners
- **Medium size image**: ~500MB, includes necessary tools, compatible with most actions (recommended)
- **Micro size image**: <200MB, only NodeJS, doesn't work with all actions

**Recommendation**: Choose "Medium" for most use cases.

## Common Issues and Troubleshooting

### Docker Not Running

**Error**: `Cannot connect to the Docker daemon`

**Solution**:
```bash
# Start Docker service
sudo service docker start

# Or if using Docker Desktop, ensure it's running
```

### Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker
```

### Package Installation Errors

**Error**: `Package 'libgl1-mesa-glx' has no installation candidate`

This occurs because `act` uses Ubuntu 24.04, which doesn't have the older `libgl1-mesa-glx` package.

**Solution**: The workflows have been updated to use `libgl1` instead, which is compatible with Ubuntu 24.04. If you encounter similar package errors:

1. Check the Ubuntu version in the error message
2. Update package names to match the Ubuntu version
3. Common replacements:
   - `libgl1-mesa-glx` → `libgl1` (Ubuntu 24.04+)
   - `libglib2.0-0` → `libglib2.0-0` (still works, but may show note about `libglib2.0-0t64`)

### Workflow Fails Locally but Works on GitHub

Some GitHub Actions features don't work locally:

- **GitHub API calls**: Actions that interact with GitHub API won't work
- **Secrets**: Need to be set manually or via environment variables
- **PR comments**: Won't post comments to actual PRs
- **Artifacts**: May behave differently
- **Package versions**: Local Docker images may use different Ubuntu versions than GitHub runners

**Workaround**: Use `-n` (dry-run) to validate structure, then test on GitHub for full functionality.

### Slow First Run

The first run downloads Docker images (~500MB for medium image). Subsequent runs are faster.

**Solution**: Images are cached, so only the first run is slow.

### Matrix Jobs Not Running

Matrix jobs may not all run in dry-run mode. This is normal.

**Solution**: Run without `-n` to see all matrix combinations execute.

### Codecov Upload Errors

**Error**: `Rate limit reached. Please upload with the Codecov repository upload token`

This is **expected** when running locally. Codecov requires authentication tokens that aren't available in local runs.

**Solution**: This is harmless - the workflow has `fail_ci_if_error: false` for Codecov, so it won't fail the build. Coverage reports are still generated locally in `htmlcov/` and `coverage.xml`.

### Artifact Upload Failures

**Error**: `Job 'Code Quality Check' failed` or artifact upload errors like:
- `Unable to get ACTIONS_RUNTIME_TOKEN env variable`
- `Error uploading artifact`

This occurs when workflows try to upload artifacts using `actions/upload-artifact@v3` or similar actions.

**Why it fails**:
- GitHub Actions artifact API isn't available locally
- `ACTIONS_RUNTIME_TOKEN` is a GitHub-specific environment variable not available locally
- Artifact upload actions don't work the same way in `act`
- Files may not exist if previous steps had `continue-on-error: true`

**Solution**: This is **expected and harmless** when running locally. The security checks (bandit, safety) still run and generate reports locally, but the upload step fails. On GitHub, artifact uploads work correctly.

**Note**: Even if the job shows as "failed" due to artifact upload errors, if your tests passed (e.g., "157 passed"), your code is working correctly. The failure is only in the artifact upload step, not in your actual tests or code.

**Workaround**: If you need the reports locally, check the generated files (e.g., `bandit-report.json`) in your working directory after the workflow runs.

### Final Cleanup Errors

**Error**: `Error occurred running finally: exitcode '1': failure`

This can occur when running workflows locally, often from:
- Conditional jobs that don't match local event context
- Cleanup steps that expect GitHub-specific environment
- Jobs with conditions like `if: github.event_name == 'push'`
- Artifact upload steps (see above)

**Solution**: This is usually harmless if your tests passed. The important jobs (tests, linting) should complete successfully. The error is typically from cleanup/finally blocks that don't affect test results.

## Best Practices

### 1. Always Dry-Run First

```bash
# Always use -n first to validate
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

### 2. Test Workflow Changes Locally

Before pushing workflow changes:

```bash
# Test the modified workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

### 3. Test Specific Jobs During Development

When developing, test only relevant jobs:

```bash
# Test only linting while fixing code style
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint
```

### 4. Use Verbose Mode for Debugging

When workflows fail, use verbose mode:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v
```

### 5. Validate Workflow Syntax

Use dry-run to catch syntax errors:

```bash
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n
```

## Workflow-Specific Notes

### CI Workflow

- Tests multiple Python versions (3.9, 3.10, 3.11)
- Runs test matrix across all test suites
- Includes linting and code quality checks
- Performance tests only run on main branch pushes (won't run in local test)

### PR Workflow

- PR number input is required
- Comments won't be posted to actual PRs
- Use for validating workflow structure

### Nightly Workflow

- Can simulate schedule trigger
- Includes security checks (bandit, safety)
- Generates security reports

### Release Workflow

- Requires version input for manual trigger
- Documentation build only runs on `release` events (not `workflow_dispatch`)
- Similar to CI workflow - runs full test suite and generates reports
- Artifact upload may fail locally (expected)

## Quick Reference

```bash
# List all workflows
act -l

# Dry-run CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -n

# Run CI workflow
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main

# Test specific job
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -j lint

# Verbose output
act workflow_dispatch -W .github/workflows/ci.yml --input branch=main -v

# Test PR workflow
act workflow_dispatch -W .github/workflows/pr.yml --input pr_number=1 --input branch=main -n

# Test nightly workflow
act schedule -W .github/workflows/nightly.yml -n

# Test release workflow (dry-run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0 -n

# Test release workflow (full run)
act workflow_dispatch -W .github/workflows/release.yml --input version=v1.0.0
```

## Additional Resources

- [act Documentation](https://github.com/nektos/act)
- [act Usage Guide](https://nektosact.com/usage/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Documentation](https://docs.docker.com/)

## Summary

Using `act` for local CI/CD testing provides:

✅ **Faster feedback** - Test workflows without pushing to GitHub  
✅ **Cost savings** - Save CI/CD minutes  
✅ **Better debugging** - Debug issues locally with verbose output  
✅ **Validation** - Catch workflow errors before they reach GitHub  
✅ **Development efficiency** - Test workflow changes quickly  

Remember: Always use `-n` (dry-run) first, then run actual tests when needed. Some GitHub-specific features won't work locally, but workflow structure and most steps can be validated.

