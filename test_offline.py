from __future__ import annotations

import graph
from eval import compare

class FakeLLM:

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else "SELECT 1 FROM Album"
        return type("Msg", (), {"content": text})()

def scripted(sql_responses: list[str]):
                                                                                          
    fake = FakeLLM(sql_responses + ["(final answer)"] * 5)
    graph.get_llm = lambda *a, **k: fake
    return fake

PASS, FAIL = "pass", "FAIL"
failures = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    failures += not condition
    print(f"  {PASS if condition else FAIL}  {name}{'  ' + detail if detail else ''}")

print("\n1. Validator catches a hallucinated column, and attempt 2 fixes it")
scripted([
    "SELECT Name FROM Track ORDER BY Duration DESC LIMIT 3",                   
    "SELECT Name FROM Track ORDER BY Milliseconds DESC LIMIT 3",
])
s = graph.run_agent("What are the 3 longest tracks?")
check("ends successfully", s["status"] == "success")
check("took exactly 2 attempts", s["attempts"] == 2, f"got {s['attempts']}")
check("attempt 1 failed at validation", s["history"][0]["stage"] == "validation")
check("error text names the column", "Duration" in (s["history"][0]["error"] or ""))
check("returned 3 rows", len(s["rows"]) == 3)

print("\n2. Hallucinated table is caught before touching the database")
scripted([
    "SELECT * FROM Songs LIMIT 5",
    "SELECT Name FROM Track LIMIT 5",
])
s = graph.run_agent("Show me some songs")
check("recovered", s["status"] == "success")
check("suggested the real table", "track" in (s["history"][0]["error"] or "").lower())

print("\n3. Runtime execution error also triggers a retry")
scripted([
                                                                             
    "SELECT Name FROM Track t JOIN Genre g ON g.GenreId = t.GenreId LIMIT 3",
    "SELECT t.Name FROM Track t JOIN Genre g ON g.GenreId = t.GenreId LIMIT 3",
])
s = graph.run_agent("Show track names with genres")
check("recovered on attempt 2", s["status"] == "success" and s["attempts"] == 2,
      f"status={s['status']} attempts={s['attempts']}")
check("failure was at execution", s["history"][0]["stage"] == "execution")

print("\n4. The loop gives up after max_attempts instead of running forever")
scripted(["SELECT Nope FROM Track"] * 6)
s = graph.run_agent("Something impossible", max_attempts=3)
check("status is failed", s["status"] == "failed")
check("stopped at 3 attempts", s["attempts"] == 3, f"got {s['attempts']}")
check("answer explains the failure", "could not answer" in s["answer"].lower())

print("\n5. An empty result is a valid answer, not a retry trigger")
scripted(["SELECT Name FROM Track WHERE Milliseconds > 36000000"])
s = graph.run_agent("Any tracks longer than 10 hours?")
check("status is success", s["status"] == "success")
check("only 1 attempt used", s["attempts"] == 1, f"got {s['attempts']}")
check("zero rows returned", len(s["rows"]) == 0)

print("\n6. Zero-count scalar is also treated as a real answer")
scripted(["SELECT COUNT(*) AS n FROM Customer WHERE Country = 'Antarctica'"])
s = graph.run_agent("How many customers in Antarctica?")
check("status is success", s["status"] == "success")
check("only 1 attempt used", s["attempts"] == 1)
check("count is 0", s["rows"] == [{"n": 0}], str(s["rows"]))

print("\n7. Write statements are rejected")
scripted(["DELETE FROM Customer"] * 4)
s = graph.run_agent("delete everything", max_attempts=2)
check("refused", s["status"] == "failed")
check("named as read-only violation", "read-only" in (s["history"][0]["error"] or "").lower())

print("\n8. Grading logic")
check("column split still counts as correct",
      compare([{"F": "Helena", "L": "Holy"}], [{"Name": "Helena Holy"}])[1])
check("wrong number is not correct", not compare([{"n": 59}], [{"n": 58}])[1])
check("empty == empty", compare([], [])[0])

print(f"\n{'ALL OFFLINE TESTS PASSED' if not failures else str(failures) + ' TEST(S) FAILED'}\n")
raise SystemExit(1 if failures else 0)
