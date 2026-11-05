import math
import random
import sys
from typing import Self

"""
A custom vector class implementation for educational purposes.
"""

class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = []
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)])

    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)
        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Type of the vectors not match {type(t)}")
        if len(self) != len(t):
            raise TypeError(f"Vectors must be the same dimension")
        return Vec([x - y for x, y in zip(self.elements, t.elements)])
        
    def __neg__(self) -> Self:
        raise RuntimeError("vec negation unimplemented")

    def __radd__(self, other):
        raise RuntimeError("vec _radd_ unimplemented")

    def __iadd__(self, other):
        raise RuntimeError("vec _iadd_ unimplemented")

    @staticmethod
    def zeros(n: int) -> Self:
        if n <= 0:
            raise ValueError("Dimension n must be positive (>0)")
        return Vec([0.0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        if n <= 0:
            raise ValueError("Dimension n must be positive (>0)")
        return Vec([1.0] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        if n <= 0:
            raise ValueError("Dimension n must be positive (>0)")
        return Vec([random.random() for _ in range(n)])

    def norm(self) -> float:
        return math.sqrt(sum(x ** 2 for x in self.elements))

    def mean(self) -> float:
        """Returns the arithmetic mean of the vector elements."""
        if len(self.elements) == 0:
            raise ValueError("Cannot calculate mean of an empty vector.")
        return sum(self.elements) / len(self.elements)

    def demean(self) -> "Vec":
        """Returns a new vector representing the de-meaned version of the original vector."""
        if len(self.elements) == 0:
            raise ValueError("Cannot calculate demean of an empty vector.")
        m = self.mean()
        demeaned_elements = [x - m for x in self.elements]
        return Vec(demeaned_elements)

    def std(self) -> float:
        """Returns the standard deviation of the vector elements using the demean method."""
        if len(self.elements) == 0:
            raise ValueError("Cannot calculate standard deviation of an empty vector.")
        demeaned = self.demean()
        variance = sum(x ** 2 for x in demeaned) / len(demeaned)
        return variance ** 0.5

    standard_deviation = std

"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.
(5) Test this implementation by importing the class in a sepatate python script.
(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements?
(7) Measure the performance on your machine. Check it on colab.

(8) use numpy and compare the performance.
"""

if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    #z1 = Vec.zeros(10)
    v1 = Vec([0, 1, 1.03])
    print(v1)
    v3 = 2.2 * v1
    v3 *= 5
    # v3 = 1 + v3
    print(v3)
    v2 = v1 + v3
    print(v1 + v3)
    #print(-(v1 + v3))
