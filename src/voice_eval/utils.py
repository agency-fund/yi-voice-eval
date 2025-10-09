"""Simple utility functions for testing imports"""


def greet_kannada(name: str) -> str:
    """
    Greet someone in Kannada.

    Args:
        name: Person's name

    Returns:
        Greeting string in Kannada
    """
    return f"ನಮಸ್ಕಾರ {name}! (Namaskara {name}!)"


def add_numbers(a: int, b: int) -> int:
    """Simple addition for testing."""
    return a + b
