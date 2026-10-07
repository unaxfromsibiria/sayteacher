import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)

if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
