# Domain documentation convention

Start all domain discovery at the root [CONTEXT-MAP.md](../../CONTEXT-MAP.md). It is the single canonical index for this multi-component Rust, Python, and Web repository.

- Repository-wide product intent and supported capabilities live in `README.md`.
- Durable component context lives in `<area>/CONTEXT.md` only after that file is registered in `CONTEXT-MAP.md`.
- Cross-component architectural decisions live in `docs/adr/`.
- Do not add a root `CONTEXT.md` while the context-map convention is active.
