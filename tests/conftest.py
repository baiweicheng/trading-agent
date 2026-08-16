"""Repository-wide test configuration for deterministic Hypothesis profiles."""

from __future__ import annotations

import os

from hypothesis import settings

# Hypothesis does not read the ``[tool.hypothesis]`` TOML table directly.  Keep
# the profile values declared in pyproject.toml effective for every pytest
# invocation, while allowing CI or a caller to select the registered profile.
settings.register_profile("default", max_examples=100, deadline=1000)
settings.register_profile(
    "ci",
    settings.get_profile("default"),
    max_examples=200,
    deadline=1000,
    derandomize=True,
)
_profile = os.environ.get("HYPOTHESIS_PROFILE")
if _profile is None:
    _profile = "ci" if os.environ.get("CI") else "default"
settings.load_profile(_profile)
