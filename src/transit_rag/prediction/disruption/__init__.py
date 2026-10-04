"""Disruption detection: the label RQ2's detectors are scored against.

The delay model in :mod:`transit_rag.prediction.model` answers one train at one
stop. A disruption is a property of a *line* over a *window*, and needs its own
ground truth before any detector can be compared. :mod:`labels` builds that
truth from the decided definition (``docs/08`` §3.2). Every threshold in it is a
parameter, because the definition is still pending the supervisor's
confirmation, and a change she asks for has to be a re-run rather than a rebuild.
"""
