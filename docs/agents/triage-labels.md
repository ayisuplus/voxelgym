# Triage label convention

Use exactly one tracker label from this mapping to represent the issue's triage state. The existing GitHub labels `question`, `help wanted`, and `wontfix` are preserved as semantic equivalents instead of adding duplicates.

| Triage state | GitHub label | Meaning |
| --- | --- | --- |
| Needs initial triage | `needs-triage` | The issue has not yet been classified or routed. |
| Needs information | `question` | Progress is blocked pending clarification or additional evidence. |
| Ready for an agent | `ready-for-agent` | Scope and acceptance criteria are clear enough for an agent to implement. |
| Ready for a human | `help wanted` | The issue is ready but should be handled or decided by a human. |
| Will not be fixed | `wontfix` | The repository has deliberately declined the work. |

Replace the current triage-state label when the state changes; do not accumulate multiple labels from this table. Other labels may remain to describe type, subsystem, or priority.
