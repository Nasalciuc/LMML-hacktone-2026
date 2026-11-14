# Angola - The Waterline Replay

**Flag:** `SIGMOID_MEM_KILAMBA` (in `answer.txt`)

Three runs of the same question. In `incident-01` the live tool says RESTORED (updated 13:58), but the answer says the outage is still on.
That run is the only one with a `memory.result` injected into the context: `SIGMOID_MEM_KILAMBA`, written at 10:00 and valid only until 11:00, saying water has NOT been restored.
`replay-no-memory` (same build, model, seed, same tool result, memory off) answers correctly, so the stale memory item is what flipped the answer.
The slow tool call in the incident run (1840 ms vs ~180 ms) returned the correct status and is not the cause.
