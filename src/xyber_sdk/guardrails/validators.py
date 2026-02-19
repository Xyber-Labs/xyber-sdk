from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from guardrails.types.on_fail import OnFailAction
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


class ValidationResponse(BaseModel):
    """The expected JSON response from the validation LLM."""

    is_malicious: bool = Field(
        description="True if the content violates the policy, False otherwise."
    )
    reason: str = Field(description="A brief explanation for the validation decision.")


@register_validator(name="LLM Validator", data_type="string")
class LLMValidator(Validator):
    """
    A custom Guardrails Validator that uses an LLM to check content.

    This validator is initialized with a Chat LLM and a system prompt. It uses
    the LLM to validate a given value, expecting a specific JSON response format
    to determine if the validation passes or fails. It's designed to be a
    reusable, first-class component within the Guardrails ecosystem.
    """

    _json_instruction_suffix = """
    You must respond with ONLY valid JSON in this exact format:
    {
        "is_malicious": true,
        "reason": "explanation here"
    }

    Rules:
    - Use lowercase 'true' or 'false' (not True/False)
    - Set "is_malicious" to true if content violates policy, false otherwise
    - Provide a brief reason in the "reason" field
    - Do not include any other text or formatting outside the JSON
    """

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        on_fail: str | None = OnFailAction.EXCEPTION,
        **kwargs,
    ):
        super().__init__(on_fail=on_fail, **kwargs)

        # Bind the LLM to produce structured output based on the Pydantic model.
        self.llm = llm
        self.system_prompt = self._json_instruction_suffix + system_prompt

    def validate(self, value: Any, metadata: dict) -> ValidationResult:
        """Synchronous validation is not supported for this LLM-based validator."""
        raise NotImplementedError("Use async_validate for LLM-based validation.")

    async def async_validate(self, value: Any, metadata: dict) -> ValidationResult:
        """
        Performs asynchronous validation on the given value using the configured LLM.
        """
        value = str(value)

        try:
            response: ValidationResponse = await self.llm.ainvoke(
                [SystemMessage(content=self.system_prompt), HumanMessage(content=value)]
            )

            result = PydanticOutputParser(pydantic_object=ValidationResponse).parse(
                response.content
            )
            if result.is_malicious:
                return FailResult(
                    error_message=result.reason,
                )
            else:
                return PassResult()

        except Exception as e:
            # If the LLM call fails (e.g., API error, or if the model *still*
            # fails to produce valid output), we treat it as a validation failure.
            return FailResult(
                error_message=f"The validation LLM failed with an unexpected error: {e}",
            )
