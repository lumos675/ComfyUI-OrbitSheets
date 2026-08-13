"""ComfyUI-OrbitSheets — character and location reference sheets from MiniMax-H3.

Self-contained: the only imports outside the standard library are ones ComfyUI
already provides (torch, numpy, PIL, requests) plus an optional
`comfy.model_management` call that degrades quietly when absent.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
