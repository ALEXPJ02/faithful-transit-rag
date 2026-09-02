"""Static corpus ingestion: Opal fare/policy PDFs -> chunked, cited passages.

Corpus (see docs/03-data-sources.md): Opal Fares Business Rules, Opal Terms
of Use, and the Public Transport Fares and Ticketing brochure. Chunks must
carry document title and page number — the faithfulness judge scores claims
against retrieved passages, so an unattributable chunk is unusable evidence.
"""
