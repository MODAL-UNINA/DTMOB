from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class ValueContainerError(Exception):
    """Base exception for ValueContainer errors."""

    pass


class ValueNotSetError(ValueContainerError):
    """Custom exception raised when the value is not set in ValueContainer."""

    pass


class ValueContainer(Generic[T]):
    def __init__(self, value: Optional[T] = None) -> None:
        self._value: Optional[T] = value

    def set(self, value: T) -> None:
        """Set the container's value."""
        self._value = value

    def clear(self) -> None:
        """Clear the container's value."""
        self._value = None

    def __bool__(self) -> bool:
        """Returns True if a value is set, otherwise False."""
        return self._value is not None

    def __call__(self) -> T:
        """
        Returns the contained value.
        Raises:
            ValueNotSetError: If no value is set.
        """
        if self._value is None:
            raise ValueNotSetError("Value not set")
        return self._value

    def __repr__(self) -> str:
        """Return a string representation of the contained value."""
        return f"{self.__class__.__name__}({self._value!r})"

    def __str__(self) -> str:
        """Return a user-friendly string representation of the contained value."""
        return str(self._value)
