from unittest.mock import AsyncMock, MagicMock

import pytest
from guardrails.errors import ValidationError

from xyber_sdk.guardrails.supervisor import (
    SnapshotCreationError,
    StateSnapshot,
    Supervisor,
)

# --- Fixtures ---


@pytest.fixture
def mock_graph():
    """Create a mock compiled graph."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="Response from graph")]})
    graph.aget_state = AsyncMock(
        return_value=MagicMock(config={"configurable": {}}, values={"messages": []})
    )
    graph.aupdate_state = AsyncMock()
    return graph


@pytest.fixture
def mock_graph_with_snapshot():
    """Create graph that supports state snapshots."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock()
    graph.aget_state = AsyncMock(
        return_value=MagicMock(
            config={"configurable": {"thread_id": "test"}},
            values={"messages": [MagicMock(content="Previous state")]},
        )
    )
    graph.aupdate_state = AsyncMock()
    return graph


@pytest.fixture
def input_state_dict():
    """Create input state as dict."""
    return {"messages": [MagicMock(content="Hello")]}


@pytest.fixture
def input_state_object():
    """Create input state as object with messages attribute."""
    state = MagicMock()
    state.messages = [MagicMock(content="Hello")]
    return state


# --- Happy Path Tests ---


async def test_basic_invocation_returns_result(mock_graph, input_state_dict):
    """Graph invocation without guards returns result."""
    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
    )

    assert result["result"] is not None
    assert result["rejection_message"] is None
    mock_graph.ainvoke.assert_called_once()


async def test_invocation_with_thread_id(mock_graph, input_state_dict):
    """Thread ID is passed correctly to config."""
    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
        thread_id="test-thread-123",
    )

    call_args = mock_graph.ainvoke.call_args
    config = call_args[0][1]
    assert config["configurable"]["thread_id"] == "test-thread-123"


async def test_invocation_with_custom_config(mock_graph, input_state_dict):
    """Custom config is passed through."""
    custom_config = {"custom_key": "custom_value"}

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
        config=custom_config,
    )

    call_args = mock_graph.ainvoke.call_args
    config = call_args[0][1]
    assert config["custom_key"] == "custom_value"


async def test_invocation_with_object_state(mock_graph, input_state_object):
    """Works with object-based state (has .messages attribute)."""
    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_object,
    )

    assert result["result"] is not None
    assert result["rejection_message"] is None


async def test_input_guard_validates_input(mock_graph, input_state_dict):
    """Input guard validates and transforms input."""
    input_guard = MagicMock()
    input_guard.parse = AsyncMock(return_value=MagicMock(validated_output="Validated input"))

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
        input_guard=input_guard,
    )

    input_guard.parse.assert_called_once()
    assert input_state_dict["messages"][-1].content == "Validated input"


async def test_output_guard_validates_output(mock_graph, input_state_dict):
    """Output guard validates and transforms output."""
    output_guard = MagicMock()
    output_guard.parse = AsyncMock(return_value=MagicMock(validated_output="Validated output"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
        output_guard=output_guard,
    )

    output_guard.parse.assert_called_once()
    assert result["result"]["messages"][-1].content == "Validated output"


async def test_snapshot_captured_before_invocation(mock_graph, input_state_dict):
    """State snapshot is captured before graph invocation."""
    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state=input_state_dict,
    )

    mock_graph.aget_state.assert_called_once()


# --- Error Handling Tests ---


async def test_empty_messages_raises_value_error(mock_graph):
    """Empty messages in input state raises ValueError."""
    with pytest.raises(ValueError, match="Input state must contain messages"):
        await Supervisor.safe_ainvoke(
            compiled_graph=mock_graph,
            input_state={"messages": []},
        )


async def test_missing_messages_raises_value_error(mock_graph):
    """Missing messages key in input state raises ValueError."""
    with pytest.raises(ValueError, match="Input state must contain messages"):
        await Supervisor.safe_ainvoke(
            compiled_graph=mock_graph,
            input_state={},
        )


async def test_graph_error_returns_generic_rejection(mock_graph):
    """Generic exception during graph execution returns generic rejection."""
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    assert result["result"] is None
    assert result["rejection_message"] == "I'm sorry, something went wrong. Please try again."


async def test_validation_error_generates_rejection_message(mock_graph):
    """ValidationError generates contextual rejection message."""
    mock_graph.ainvoke = AsyncMock(side_effect=ValidationError("Content violates policy"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Bad content")]},
    )

    assert result["result"] is None
    assert result["rejection_message"] == "I apologize, but I cannot process that request."


async def test_validation_error_with_rejection_llm(mock_graph):
    """ValidationError with rejection LLM generates custom message."""
    mock_graph.ainvoke = AsyncMock(side_effect=ValidationError("Content violates policy"))

    rejection_llm = MagicMock()
    rejection_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="I'm sorry, your request cannot be processed due to policy restrictions."
        )
    )

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Bad content")]},
        rejection_llm=rejection_llm,
        rejection_prompt="Generate a polite rejection",
    )

    assert result["result"] is None
    assert "policy restrictions" in result["rejection_message"]
    rejection_llm.ainvoke.assert_called_once()


async def test_rejection_llm_failure_returns_default_message(mock_graph):
    """If rejection LLM fails, return default rejection message."""
    mock_graph.ainvoke = AsyncMock(side_effect=ValidationError("Content violates policy"))

    rejection_llm = MagicMock()
    rejection_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Bad content")]},
        rejection_llm=rejection_llm,
        rejection_prompt="Generate a polite rejection",
    )

    assert result["rejection_message"] == "I apologize, but I cannot process that request."


async def test_input_guard_failure_triggers_rejection(mock_graph):
    """Input guard validation failure triggers rejection flow."""
    input_guard = MagicMock()
    input_guard.parse = AsyncMock(side_effect=ValidationError("Invalid input"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Bad input")]},
        input_guard=input_guard,
    )

    assert result["result"] is None
    assert result["rejection_message"] is not None
    mock_graph.ainvoke.assert_not_called()


async def test_output_guard_failure_triggers_rejection(mock_graph):
    """Output guard validation failure triggers rejection flow."""
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="Bad output")]})

    output_guard = MagicMock()
    output_guard.parse = AsyncMock(side_effect=ValidationError("Invalid output"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Hello")]},
        output_guard=output_guard,
    )

    assert result["result"] is None
    assert result["rejection_message"] is not None


# --- Rollback Tests ---


async def test_rollback_on_graph_error(mock_graph_with_snapshot):
    """State is rolled back when graph throws an error."""
    mock_graph_with_snapshot.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph_with_snapshot,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    mock_graph_with_snapshot.aupdate_state.assert_called_once()


async def test_rollback_on_validation_error(mock_graph_with_snapshot):
    """State is rolled back when validation fails."""
    mock_graph_with_snapshot.ainvoke = AsyncMock(side_effect=ValidationError("Invalid"))

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph_with_snapshot,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    mock_graph_with_snapshot.aupdate_state.assert_called_once()


async def test_rollback_uses_captured_snapshot(mock_graph_with_snapshot):
    """Rollback uses the snapshot captured before invocation."""
    mock_graph_with_snapshot.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph_with_snapshot,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    call_args = mock_graph_with_snapshot.aupdate_state.call_args
    config = call_args[0][0]
    values = call_args[0][1]

    assert config == {"configurable": {"thread_id": "test"}}
    assert "messages" in values


async def test_no_rollback_when_snapshot_fails():
    """No rollback attempted if snapshot creation fails."""
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))
    mock_graph.aget_state = AsyncMock(side_effect=RuntimeError("Cannot get state"))
    mock_graph.aupdate_state = AsyncMock()

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    mock_graph.aupdate_state.assert_not_called()


async def test_rollback_failure_continues_gracefully(mock_graph_with_snapshot):
    """If rollback fails, continue gracefully with rejection message."""
    mock_graph_with_snapshot.ainvoke = AsyncMock(side_effect=RuntimeError("Graph crashed"))
    mock_graph_with_snapshot.aupdate_state = AsyncMock(side_effect=RuntimeError("Rollback failed"))

    result = await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph_with_snapshot,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    # Should still return a rejection message despite rollback failure
    assert result["result"] is None
    assert result["rejection_message"] is not None


async def test_no_rollback_on_success(mock_graph_with_snapshot):
    """No rollback when graph executes successfully."""
    mock_graph_with_snapshot.ainvoke = AsyncMock(
        return_value={"messages": [MagicMock(content="Success")]}
    )

    await Supervisor.safe_ainvoke(
        compiled_graph=mock_graph_with_snapshot,
        input_state={"messages": [MagicMock(content="Hello")]},
    )

    mock_graph_with_snapshot.aupdate_state.assert_not_called()


# --- StateSnapshot Tests ---


def test_snapshot_stores_config_and_values():
    """StateSnapshot stores config and values."""
    config = {"key": "value"}
    values = {"messages": []}

    snapshot = StateSnapshot(config=config, values=values)

    assert snapshot.config == config
    assert snapshot.values == values


def test_snapshot_preserves_references():
    """StateSnapshot preserves object references."""
    config = {"nested": {"key": "value"}}
    values = {"messages": ["msg1", "msg2"]}

    snapshot = StateSnapshot(config=config, values=values)

    assert snapshot.config is config
    assert snapshot.values is values


# --- SnapshotCreationError Tests ---


def test_snapshot_creation_error_is_exception():
    """SnapshotCreationError is an Exception."""
    assert issubclass(SnapshotCreationError, Exception)


def test_snapshot_creation_error_can_be_raised_and_caught():
    """Can be raised and caught."""
    with pytest.raises(SnapshotCreationError):
        raise SnapshotCreationError("Failed to create snapshot")
