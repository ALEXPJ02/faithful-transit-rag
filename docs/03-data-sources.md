# Dataset & API Selection — TfNSW Open Data Hub

Capstone: Agentic RAG system for Sydney public transport Q&A with a real-time disruption layer and eval harness.

## 1. Access basics

Registration is free at opendata.transport.nsw.gov.au: create an account, add an "application," and a key is issued immediately (no approval wait). Default tier is the **Bronze Plan** — 60,000 calls/day, 5 requests/second, shared across all TfNSW APIs called with that key. Auth is a single `Authorization` header. This comfortably covers a semester project: even polling every real-time feed every 15–30s for one Sydney area (Central–Chatswood corridor) stays well under the daily quota.

The portal hosts 1000+ resources total, organised by mode (bus, train, ferry, light rail, metro) and by static vs. real-time.

## 2. Candidate sources

| Source | Type | What it gives you |
|---|---|---|
| **Timetables Complete GTFS** | Static, all operators | Full schedule bundle: `stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `shapes.txt`, `calendar.txt`. Includes regional/trackwork routes not in real-time feeds. |
| **Public Transport – Timetables – For Realtime** | Static, real-time-capable operators only | Same GTFS files, scoped to the operators you can pair with live data — this is the one to actually use for joins. |
| **Realtime Trip Update** (per mode) | GTFS-R (protobuf) | Live stop-time updates for active trips — delays, skipped stops. |
| **Realtime Vehicle Positions** (per mode) | GTFS-R (protobuf) | Live vehicle lat/long, updated every 15s. |
| **Realtime Alerts v2** (per mode) | GTFS-R (protobuf) | Stop/trip/route-level disruption alerts, each with a natural-language `header_text` / `description_text` — the closest thing to "live text" in the real-time layer. |
| **Trip Planner API** (Stop Finder, Trip Planner, Departure, Service Alert, Coordinate Request) | REST/JSON | Full trip-planning: multi-leg journeys with walking/driving legs, real-time-aware, and **includes Opal fare in the response**. This effectively replaces having to hand-roll trip logic over raw GTFS. |
| **Opal Fares** dataset | Structured (CSV/JSON) | Distance-band fare tables per mode, concession rules, caps. |
| **Regional Bus Fares** dataset | Structured | Fare bands/sections for regional bus. |
| **Opal Fares Business Rules and Information** (PDF) | Unstructured text | Prose explanation of how fares/caps/concessions are actually calculated — genuine policy document. |
| **Opal Terms of Use** (PDF) | Unstructured text | Legal/conditions-of-use document — good "policy Q&A" test case (e.g. "can I get a refund if my train is delayed?"). |
| **Public Transport Fares and Ticketing brochure** (PDF) | Unstructured text | Consumer-facing fares explainer, shorter and more readable than the business-rules PDF. |
| Live Traffic Hazards/Cameras, Loading Zones, Off-street Parking | Static/real-time | Roads/parking data — out of scope for a public-transport RAG use case. |

## 3. Scoring against your priorities

| Source | RAG suitability | Data richness | Ease of access |
|---|---|---|---|
| Static GTFS (Timetables Complete / For Realtime) | Low (tabular, not prose) | High | High — plain download or API |
| GTFS-R Trip Update / Vehicle Positions | Low (binary positional data) | High, updates every 15s | High, but needs protobuf decoding (`gtfs-realtime-bindings`) |
| GTFS-R Alerts v2 | **Medium** — has real natural-language text, but short and templated | Medium | Same as above |
| Trip Planner API | Low-medium (structured JSON, not prose) | High — one call gets route+fare+alerts | High — plain REST/JSON, no protobuf |
| Opal Fares / Regional Bus Fares (structured) | Low | Medium | High |
| Opal Fares Business Rules (PDF) | **High** — genuine policy prose | Medium | High — static download |
| Opal Terms of Use (PDF) | **High** — genuine legal/policy prose | Medium | High — static download |
| Fares & Ticketing brochure (PDF) | **High** — consumer-friendly prose | Low-medium | High — static download |

## 4. Recommendation

Don't pick one source — the project needs two distinct layers, and TfNSW's hub happens to cleanly separate them:

**RAG corpus (retrieval over static text):** Opal Fares Business Rules and Information, Opal Terms of Use, and the Fares & Ticketing brochure (all PDF). These three give a real, citable policy corpus for the "faithfulness/hallucination" eval questions (fares, refunds, concessions, conditions of travel) — this is the part of the RAG loop your eval harness will score.

**Real-time tool layer (MCP tool calls, not retrieval):** GTFS-Realtime Alerts v2 + Trip Update + Vehicle Positions for bus/train/ferry/light rail, joined against the **Public Transport – Timetables – For Realtime** static bundle for stop/route names. This is what answers "get me from Central to Chatswood accounting for current disruptions."

**Skip for now:** the raw Trip Planner API as a *replacement* for hand-rolled GTFS joins — it's convenient, but doing the join yourself (static GTFS + GTFS-R) is better portfolio signal per your stack notes, and keeps the "agent reasoning over tool outputs" story intact rather than outsourcing trip logic to TfNSW's own planner. Keep it in your back pocket as a fallback if GTFS joins prove too time-consuming given the compressed timeline. Roads/parking data: out of scope, drop entirely.

## 5. Notes / risks

- GTFS-R responses are protobuf, not JSON — budget time for the `gtfs-realtime-bindings` decode step early, since it's a common first-week snag.
- The Alerts v2 `description_text` field is a secondary, smaller source of "real" natural-language text you could optionally fold into the RAG index alongside the PDFs, since it's the one piece of real-time content that reads like prose rather than data.
- PDFs update occasionally (Opal Terms of Use had an April 2026 and June 2026 revision within this year) — pin a version/date when building the corpus so eval results are reproducible.
- Most datasets are Creative Commons BY 4.0 — safe to use in a public repo/live demo, which matters for the portfolio angle (check the licence tag per-dataset, it's not blanket across the whole hub).

## 6. The historical-data gap (added 2026-08-23)

The Open Data Hub has **no usable historical archive for bus or train GTFS-Realtime
data**. The "Historical GTFS and GTFS Realtime" API has covered Metro and Ferry only
since 2020; public requests for bus/train history (2021–2025, still unanswered as of
a May 2026 forum thread) are consistently met with "collect the live feed going
forward". Aggregate on-time-running datasets go back to 2010 but are monthly
percentage rollups — no per-trip records, so nothing a model can train on.

This is not a minor sourcing inconvenience. It determines the shape of the whole
prediction layer: the model can only train on data collected during the project, so
the collector's start date sets a hard ceiling on the training set, and its uptime is
the project's critical path. See [`02-tech-stack.md`](./02-tech-stack.md) §3 and
[`04-implementation-plan.md`](./04-implementation-plan.md) §1.
