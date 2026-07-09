class CriticError(Exception):
    """Base exception for all Critic Agent errors."""
    pass

class LLMInvocationError(CriticError):
    """Raised when calling the LLM fails."""
    pass

class ParserError(CriticError):
    """Raised when parsing or validating the LLM output fails."""
    pass

class BusinessRuleError(CriticError):
    """Raised when business rule validation fails or encounters issues."""
    pass
