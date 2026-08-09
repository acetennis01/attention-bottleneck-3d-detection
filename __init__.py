"""Custom camera-LiDAR fusion project.

Model and dataset registration is intentionally explicit through each config's
``custom_imports`` entry. Eager imports here cause duplicate MMEngine registry
entries when tools load this project under different package roots.
"""

__all__ = []
