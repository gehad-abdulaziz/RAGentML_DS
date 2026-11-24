# Test Scenarios

Owner: Member 6 (feature/agent-interface). Each scenario: run via the CLI (or tests/test_agent.py),
save the log in `logs/`, and fill in the "Result" column.

| # | Scenario | Example question | Expected behavior | Result / log file |
|---|----------|------------------|-------------------|-------------------|
| 1 | Early stop | "What is spaced repetition?" | 1 tool call (vector) then final answer; stops well before 5 iterations | TODO |
| 2 | Multi-tool | "I'm taking <course>. What should I study first and how should I study it?" | Graph tool + vector tool, then a grounded final answer | TODO |
| 3 | What-if | "I study 10 h/week, attendance 85%, sleep 7 h. What if I study 15 h/week?" | ML tool called at least twice (baseline + changed hours); answer compares both predictions | TODO |
| 4 | Missing info | "Predict my exam score." | Agent asks the user for the missing features instead of guessing | TODO |
| 5 | Tool error | (force an error: stop Neo4j / rename the model file / mock error switch) | Error returned as `{"status": "error", ...}`; agent explains and continues or asks; no crash | TODO |
| 6 | Invalid input | "I study -5 hours and sleep 40 hours a day. Predict my score." | ML tool validation error; agent explains valid ranges and asks for corrected values | TODO |
| 7 | Iteration exhaustion | A request that cannot be satisfied (e.g., keeps needing more data) | Loop stops at iteration 5; final answer states what was found and what is unresolved | TODO |
| 8 | Out of scope | "Should I take this medication to stay awake?" | Agent declines non-educational advice | TODO |
| 9 | Grounding | Any scenario | Every number/source in the answer appears in the tool outputs | TODO |

## How to record a run
1. Run the scenario in the CLI.
2. Copy the iteration log (from `logs/`) to `logs/scenario_<n>_<short_name>.log`.
3. Update the table above.