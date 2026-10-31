"""The one regular-time-axis implementation.

Every store gets its time axis from here, so cadence, gaps and UTC handling
are decided once rather than per source.

Not implemented yet — see the Target design tab of the redesign plan.
"""
from __future__ import annotations
