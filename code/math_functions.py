# math_functions.py


def add(a: int, b: int) -> int:
    """Adds two integers and returns the result"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Substracts two integers and returns the result"""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result"""
    return a * b


def divide(a: int, b: int) -> float:
    """Divides two integers and returns the result"""
    result = a / b
    return round(result, 2)


def power(a: int, b: int) -> int:
    """Raises a to the power of b and returns the result"""
    return a**b


def square_root(a: float) -> float:
    """Returns the square root of a"""
    return a**0.5
