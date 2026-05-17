---
name: workgraph
description: Start WorkGraph listening from inside Codex, backfill the current session into WorkGraph, inspect compiled session memory, reuse exported skills, and export new skills from agent sessions.
---

# WorkGraph

Use this skill when the user asks to start WorkGraph, listen to this session, connect Codex to WorkGraph, summarize the session so far, reuse a previous technique, export a reusable pattern, or inspect the WorkGraph UI.

## Commands To Run

- Start listening or backfill this session: `workgraph start codex`
- Stop listening: `workgraph stop .`
- Status: `workgraph status .`
- Reuse a pattern: `workgraph reuse "<query>"`
- Export a skill: `workgraph skill export "<name>" . --query "<query>"`

After starting, tell the user WorkGraph is listening and the UI is at `http://127.0.0.1:8787`. The first watch pass ingests the latest Codex session history for this repo, so starting mid-session is valid.
