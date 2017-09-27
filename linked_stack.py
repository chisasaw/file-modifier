from node import LinkedList
from container import Container


class Stack(Container):
    """
    Last-in, first-out (LIFO) stack.
    """

    def __init__(self):
        """
        Create a new, empty Stack self.

        Overrides Container.__init__

        @param Stack self: this stack
        @rtype: None
        """
        self._list = LinkedList()

    def add(self, obj):
        """
        Add object obj to top of Stack self.

        @param Stack self: this Stack
        @param object obj: object to place on Stack
        @rtype: None
        """
        self._list.prepend(obj)

    def remove(self):
        """
        Remove and return top element of Stack self.

        Assume Stack self is not empty.

        @param Stack self: this Stack
        @rtype: object

        >>> s = Stack()
        >>> s.add(5)
        >>> s.add(7)
        >>> s.remove()
        7
        """
        return self._list.delete_front()

    def is_empty(self):
        """
        Return whether Stack self is empty.

        @param Stack self: this Stack
        @rtype: bool
        """
        return self._list.size == 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()
