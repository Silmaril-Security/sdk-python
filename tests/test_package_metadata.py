from importlib.metadata import metadata

from packaging.requirements import Requirement
from packaging.version import Version


def _langchain_core_extra_requirement() -> Requirement:
    requirements = metadata("silmaril-security-sdk").get_all("Requires-Dist") or []
    matches = []
    for value in requirements:
        requirement = Requirement(value)
        if (
            requirement.name == "langchain-core"
            and requirement.marker is not None
            and requirement.marker.evaluate({"extra": "langchain"})
        ):
            matches.append(requirement)
    assert len(matches) == 1
    return matches[0]


def test_langchain_extra_secure_union_truth_table():
    specifier = _langchain_core_extra_requirement().specifier
    expected = {
        "0.3.84": False,
        "0.3.85": True,
        "1.0.0": False,
        "1.0.99": False,
        "1.1.0": False,
        "1.1.8": False,
        "1.2.0": False,
        "1.2.99": False,
        "1.3.0": False,
        "1.3.1": False,
        "1.3.2": False,
        "1.3.3": True,
        "1.7.6": True,
    }
    assert {
        version: Version(version) in specifier
        for version in expected
    } == expected
