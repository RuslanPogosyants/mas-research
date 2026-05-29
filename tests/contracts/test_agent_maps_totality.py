"""Contract: the four agent maps stay mutually consistent and total over Operation.

OPERATION_TO_AGENT, AGENT_CLASS_NAMES, payload _BUILDERS, and DEPENDENCY_MAP must
agree so that adding F3-F6 cannot silently desync routing/assembly/dispatch.
"""

from __future__ import annotations

from src.agents.payloads import _BUILDERS
from src.core.schemas import Operation
from src.plan import AGENT_CLASS_NAMES, DEPENDENCY_MAP, OPERATION_TO_AGENT


def test_every_operation_has_an_agent() -> None:
    assert set(OPERATION_TO_AGENT) == set(Operation)


def test_every_agent_has_a_class_name() -> None:
    assert set(AGENT_CLASS_NAMES) == set(OPERATION_TO_AGENT.values())


def test_every_operation_has_a_payload_builder() -> None:
    assert set(_BUILDERS) == set(Operation)


def test_dependency_map_keys_and_values_are_valid_operations() -> None:
    for operation, upstreams in DEPENDENCY_MAP.items():
        assert operation in Operation
        for upstream in upstreams:
            assert upstream in Operation
