# Security Policy

## Supported Versions

Security fixes are prepared for the latest published minor release line of the
Python SDK. Users should upgrade to the newest PyPI release unless Silmaril
support has directed them to a pinned version.

## Reporting a Vulnerability

Do not open public GitHub issues for vulnerabilities, leaked credentials, or
tenant-specific security details.

Report security concerns through GitHub private vulnerability reporting if it is
enabled for this repository, or through your existing private Silmaril support
channel. Include:

- Affected SDK version and Python version
- Impacted integration surface, such as the core client, async client, LangChain
  handler, Deep Agents middleware, governance, retries, packaging, or release
  automation
- Reproduction steps or a minimal proof of concept
- Whether any credentials, tenant IDs, endpoint URLs, or customer data were
  exposed

Silmaril will acknowledge reports, triage severity, and coordinate remediation
privately. Public disclosure should wait until a fix or mitigation is available.

## Secrets and Test Data

Never commit API keys, tenant endpoint URLs, `.env` files, generated
distributions, or live customer payloads. This repository has no live
integration suite. A test that calls a deployed Firewall must skip unless the
caller opts in, and must be marked `@pytest.mark.integration`. Pull-request CI
runs `pytest -q` with no marker filter; the release workflow skips that marker.
