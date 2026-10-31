"""Grid registry: id -> CRS, shape, transform; regrid(src_id, dst_id).

Holds the KNMI 1 km stereographic product grid (700x765, corners from
geo_product_corners) and every other named grid in configs/grids.yaml.

Not implemented yet — see the Target design tab of the redesign plan.
"""
from __future__ import annotations
