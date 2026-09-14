"""Live TfNSW GTFS-Realtime access.

Shared by two consumers with different lifecycles: the unattended delay
collector (``transit_rag.prediction.collection``) and the agent's live tool
layer (``transit_rag.mcp_server``). Fetching and parsing live here so those
two never drift apart.
"""
