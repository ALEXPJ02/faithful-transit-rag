"""Phase 1b — the bounded-accuracy delay-prediction layer.

TfNSW publishes no usable historical GTFS-Realtime archive for bus or train
services, so the model trains on data this project collects itself. That
constraint shapes the whole package: ``collection`` runs from day one and
must keep running, ``features`` and ``model`` only matter once a collection
window has closed. See docs/02-tech-stack.md.
"""
