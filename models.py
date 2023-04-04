"""
Morphological models for astrophysical sources.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.modeling import Parameter, Fittable2DModel

__all__ = [
    'Delta2D',
]


class Delta2D(Fittable2DModel):
    """Two dimensional delta function .

    This model can be used for a point source morphology.

    Parameters
    ----------
    amplitude : float
        Peak value of the point source
    x_0 : float
        x position center of the point source
    y_0 : float
        y position center of the point source

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = \\cdot \\left \\{
                    \\begin{array}{ll}
                        A & :  x = x_0 \\ \\mathrm{and} \\ y = y_0 \\\\
                        0 & : else
                    \\end{array}
                \\right.
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, amplitude, x_0, y_0, **constraints):
        super(Delta2D, self).__init__(amplitude=amplitude, x_0=x_0,
                                      y_0=y_0, **constraints)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0):
        """Two dimensional delta model function using a local rectangular pixel
        approximation.
        """
        # _, grad_x = np.gradient(x)
        # grad_y, _ = np.gradient(y)
        # x_diff = np.abs((x - x_0) / grad_x)
        # y_diff = np.abs((y - y_0) / grad_y)

        if x.ndim == 1 & y.ndim == 1:
            n = int(np.sqrt(x.shape[0]))
            x = x.reshape(n, n)
            y = y.reshape(n, n)
            _, grad_x = np.gradient(x)
            grad_y, _ = np.gradient(y)
            x_diff = np.abs((x - x_0) / grad_x)
            y_diff = np.abs((y - y_0) / grad_y)

            x_val = np.select([x_diff < 1], [1 - x_diff], 0)
            y_val = np.select([y_diff < 1], [1 - y_diff], 0)

            return x_val * y_val * float(amplitude)
        else:
            _, grad_x = np.gradient(x)
            grad_y, _ = np.gradient(y)
            x_diff = np.abs((x - x_0) / grad_x)
            y_diff = np.abs((y - y_0) / grad_y)

            x_val = np.select([x_diff < 1], [1 - x_diff], 0)
            y_val = np.select([y_diff < 1], [1 - y_diff], 0)

        return x_val * y_val * float(amplitude)