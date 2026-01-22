import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from guardrails.errors import ValidationError

logger = logging.getLogger(__name__)


class SnapshotCreationError(Exception):
    """
    Exception raised when a snapshot of the current state of the graph cannot be created.
    """

    pass


class StateSnapshot:
    """
    A class that represents a snapshot of the current state of the graph.
    """

    def __init__(self, config: dict, values: dict):
        self.config = config
        self.values = values


class Supervisor:
    """
    A supervisor class that provides safe graph execution with input and output validation.
    """

    @staticmethod
    async def safe_ainvoke(
        compiled_graph,
        input_state: Any,
        input_guard: Any | None = None,
        output_guard: Any | None = None,
        rejection_llm: BaseChatModel | None = None,
        rejection_prompt: str | None = None,
        config: dict | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Safely invoke a compiled graph with input and output validation.

        Args:
            compiled_graph: The compiled LangGraph instance
            input_state: The input state to pass to the graph. Must have a `messages` attribute
                        (list of messages) or be a dict with a "messages" key.
            input_guard: The Guardrails guard to validate the input
            output_guard: The Guardrails guard to validate the output
            rejection_llm: The language model to generate a rejection message
            rejection_prompt: The prompt to use for the rejection LLM
            config: Optional configuration dict for the graph invocation
            thread_id: Optional thread ID for state persistence

        Returns:
            The final state if successful, or a rejection message if validation fails
        """
        invoke_config = config or {}
        if thread_id:
            invoke_config.setdefault("configurable", {})["thread_id"] = thread_id

        # 0. Check if messages are present in the input state
        messages = getattr(input_state, "messages", None) or input_state.get("messages", [])
        if not messages:
            logger.error("Supervisor: Input state must contain messages")
            raise ValueError("Input state must contain messages")

        # 1. Get the snapshot of the current state of the graph
        try:
            snapshot_before_invoke = await Supervisor._get_state_snapshot(
                compiled_graph, invoke_config
            )
        except SnapshotCreationError as e:
            logger.warning(
                f"Supervisor: Failed to get current state for rollback, proceeding without rollback capability: {e}"
            )
            snapshot_before_invoke = None

        # 2. Validate the input
        try:
            if input_guard:
                # Get the last message content
                if hasattr(input_state, "messages"):
                    last_message = input_state.messages[-1]
                    validated_input = await input_guard.parse(last_message.content)
                    input_state.messages[-1].content = validated_input.validated_output
                else:
                    last_message = input_state["messages"][-1]
                    validated_input = await input_guard.parse(last_message.content)
                    input_state["messages"][-1].content = validated_input.validated_output

            # Attempt the graph invocation
            result = await compiled_graph.ainvoke(input_state, invoke_config)

            if output_guard:
                # LangGraph returns dict results
                if isinstance(result, dict) and "messages" in result:
                    validated_output = await output_guard.parse(result["messages"][-1].content)
                    result["messages"][-1].content = validated_output.validated_output
                elif hasattr(result, "messages"):
                    validated_output = await output_guard.parse(result.messages[-1].content)
                    result.messages[-1].content = validated_output.validated_output

            return {"result": result, "rejection_message": None}

        # 3. Handle any invocation error
        except Exception as e:
            logger.error(
                f"Supervisor caught an error during graph invocation: {type(e).__name__}: {e}",
                exc_info=True,  # Log traceback for unexpected errors
            )
            if snapshot_before_invoke:
                try:
                    # Rollback to the captured state before invoke
                    await compiled_graph.aupdate_state(
                        snapshot_before_invoke.config,
                        snapshot_before_invoke.values,
                    )
                    logger.info(f"🔄 State rolled back due to error: {e}")

                except Exception as rollback_error:
                    logger.error(
                        f"❌ Failed to rollback state: {rollback_error}, proceeding without rollback"
                    )

            # For known validation errors, generate a polite, contextual rejection.
            # For unexpected errors, provide a generic failure message.
            if isinstance(e, (ValidationError, ValueError)):
                rejection_message = await Supervisor._generate_rejection_message(
                    rejection_llm, rejection_prompt, e
                )
                if hasattr(input_state, "messages") and input_state.messages:
                    last_message_content = input_state.messages[-1].content
                elif isinstance(input_state, dict) and input_state.get("messages"):
                    last_message_content = input_state["messages"][-1].content
                else:
                    last_message_content = ""
                logger.warning(
                    f"Validation error for prompt: '{last_message_content}'. Rejection: '{rejection_message}'"
                )
            else:
                rejection_message = "I'm sorry, something went wrong. Please try again."

            return {"result": None, "rejection_message": rejection_message}

    @staticmethod
    async def _generate_rejection_message(
        rejection_llm: BaseChatModel | None, rejection_prompt: str | None, e: ValidationError
    ) -> str:
        """
        Generate a rejection message for a given ValidationError.
        """

        if rejection_llm and rejection_prompt:
            try:
                logger.info("Using rejection LLM for rejection message generation")
                rejection_message = await rejection_llm.ainvoke(
                    [
                        SystemMessage(content=rejection_prompt),
                        HumanMessage(
                            content=f"Content was rejected by guardrails. Reason: {e}. Please provide a polite rejection message."
                        ),
                    ]
                )

                return rejection_message.content

            except Exception as rejection_llm_error:
                logger.error(f"Error generating rejection message: {rejection_llm_error}")
                return "I apologize, but I cannot process that request."

        return "I apologize, but I cannot process that request."

    @staticmethod
    async def _get_state_snapshot(compiled_graph, invoke_config) -> StateSnapshot:
        """
        Get a snapshot of the current state of the graph.
        """
        # Capture the current state before invocation for potential rollback
        snapshot_before_invoke = None
        try:
            current_state = await compiled_graph.aget_state(invoke_config)
            if current_state:
                snapshot_before_invoke = StateSnapshot(
                    config=current_state.config,
                    values=current_state.values,
                )
            return snapshot_before_invoke

        except Exception:
            logger.warning(
                "Supervisor: Failed to get current state for rollback, proceeding without rollback capability"
            )
            raise SnapshotCreationError
