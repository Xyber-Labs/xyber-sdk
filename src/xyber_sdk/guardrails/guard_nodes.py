from abc import ABC, abstractmethod

from langgraph.graph import END, START, StateGraph
from langgraph.typing import StateT

from guardrails import AsyncGuard
from guardrails.errors import ValidationError


class CallableNode(ABC):
    """Abstract base class for callable graph nodes."""

    @abstractmethod
    async def __call__(self, state: StateT) -> dict: ...


class GuardrailsNode(CallableNode):
    """
    A validation node that integrates Guardrails into a LangGraph.
    - On success, it updates the last message with the validated output.
    - On failure, it raises a ValidationError to halt execution.
    """

    def __init__(self, guard: AsyncGuard, messages_key: str = "messages"):
        self.name = guard.name
        self.guard = guard
        self.messages_key = messages_key

    async def __call__(self, state: StateT) -> dict:
        # Support both dict-like (TypedDict) and attribute access (BaseModel/dataclass)
        if isinstance(state, dict):
            messages = state.get(self.messages_key, [])
        else:
            messages = getattr(state, self.messages_key, [])
        if not messages:
            return {}

        last_message = messages[-1]

        try:
            outcome = await self.guard.parse(last_message.content)

            # --- Success Path ---
            msg_dict = last_message.model_dump()
            msg_dict["content"] = outcome.validated_output
            updated_message = last_message.__class__(**msg_dict)
            return {self.messages_key: [updated_message]}

        # --- Failure Path ---
        except ValidationError as e:
            raise ValidationError(f"Validation failed in node '{self.name}': {e}") from e


class GuardInjector:
    """A utility to inject GuardrailsNode into a StateGraph."""

    @staticmethod
    def _place_node_between(
        graph: StateGraph,
        node_callable: CallableNode,
        node_name: str,
        source_node_name: str,
        target_node_name: str,
    ) -> None:
        """
        Places a node between two existing nodes in the graph.
        DOES NOT WORK FOR CONDITIONAL EDGES
        """

        # Reroute: source_node -> node_name -> target_node

        graph.add_node(node_name, node_callable)

        # Remove the edge from source_node to target_node
        source, target = next(
            (s, t) for s, t in graph.edges if s == source_node_name and t == target_node_name
        )
        graph.edges.remove((source, target))

        # Add the edge from source_node to node_name
        graph.add_edge(source, node_name)

        # Add the edge from node_name to target_node
        graph.add_edge(node_name, target)

    @staticmethod
    def _delete_node(graph: StateGraph, node_name: str) -> None:
        """
        Deletes a node from the graph, сonnect it's parents to it's children
        DOES NOT WORK FOR CONDITIONAL EDGES
        """
        if node_name not in graph.nodes:
            return

        # Connect it's parents to it's children
        parent = next((s for s, t in graph.edges if t == node_name), None)
        child = next((t for s, t in graph.edges if s == node_name), None)

        if parent and child:
            graph.add_edge(parent, child)

        # Remove it's edges
        edges_to_remove = [(s, t) for s, t in graph.edges if s == node_name or t == node_name]
        for edge in edges_to_remove:
            graph.edges.remove(edge)

        # Remove the node
        graph.nodes.pop(node_name)

    @staticmethod
    def _reroute_conditional_edges(obj: object, original_dest: object, new_dest: object) -> None:
        """
        Recursively traverse any nested structure (dicts, lists, objects with 'ends' attribute, etc.)
        and replace all values equal to original_dest with new_dest, in-place.
        """
        # Handle dict-like
        if isinstance(obj, dict):
            for k, v in obj.items():
                if v == original_dest:
                    obj[k] = new_dest
                else:
                    GuardInjector._reroute_conditional_edges(v, original_dest, new_dest)
        # Handle list/tuple/set
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                GuardInjector._reroute_conditional_edges(item, original_dest, new_dest)
        # Handle objects with 'ends' attribute (like Branch)
        elif hasattr(obj, "ends") and isinstance(getattr(obj, "ends"), dict):
            ends = getattr(obj, "ends")
            for k, v in ends.items():
                if v == original_dest:
                    ends[k] = new_dest
                else:
                    GuardInjector._reroute_conditional_edges(v, original_dest, new_dest)
        # Optionally, handle objects with __dict__ (custom classes)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                GuardInjector._reroute_conditional_edges(v, original_dest, new_dest)

    @staticmethod
    def inject_input_validator(graph: StateGraph, validator: CallableNode) -> None:
        """
        Injects an input validator node right after the graph's entry point.

        Args:
            graph: The StateGraph to modify.
            validator: The ValidatorNode to inject.
        """

        parent_name, child_name = next((s, t) for s, t in graph.edges if s == START)

        GuardInjector._place_node_between(
            graph,
            validator,
            validator.name + "_input_validator",
            parent_name,
            child_name,
        )

    @staticmethod
    def inject_tool_validator(
        graph: StateGraph,
        validator: CallableNode,
        tool_node_name: str = "tools",
    ) -> None:
        """
        Injects a tool validator node after tool execution but before returning to LLM.
        This ensures tools can be called freely, but their results are validated
        before being processed by the LLM.

        Args:
            graph: The StateGraph to modify.
            validator: The ValidatorNode to inject.
            tool_node_name: Optional override for the name of the tool node.
        """

        parent_name, child_name = next((s, t) for s, t in graph.edges if s == tool_node_name)

        GuardInjector._place_node_between(
            graph,
            validator,
            validator.name + "_tool_validator",
            parent_name,
            child_name,
        )

    @staticmethod
    def inject_output_validator(graph: StateGraph, validator: CallableNode) -> None:
        """
        Injects an output validator node before any branch of the graph terminates.
        Args:
            graph: The StateGraph to modify.
            validator: The ValidatorNode to inject.
        """

        validation_node_name = validator.name + "_output_validator"
        graph.add_node(validation_node_name, validator)

        # Reroute all edges that originally pointed to the end
        edges_to_end = [edge for edge in graph.edges if edge[1] == END]
        for source, _ in edges_to_end:
            graph.edges.remove((source, END))
            graph.add_edge(source, validation_node_name)

        # Reroute all conditional branches that originally pointed to the end (robust, recursive)
        GuardInjector._reroute_conditional_edges(graph.branches, END, validation_node_name)

        # From the output validator, straight to the end
        graph.add_edge(validation_node_name, END)
