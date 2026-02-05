"""
TEMPORARY COMPATIBILITY SHIM
This file exists ONLY to fix Render's deployment issue.
Render is somehow deploying old code that imports from this file.
This shim redirects to the correct core_features module.

TODO: Remove this file once Render is properly deploying from main branch.
"""

from app.audio.core_features import extract_features

__all__ = ['extract_features']
