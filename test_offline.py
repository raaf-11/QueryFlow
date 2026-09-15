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

print("\n8. Cross-check catches a wrong answer that no error would reveal")

scripted([
    "SELECT Name FROM Track WHERE Milliseconds > 36000",                          
    "SELECT Name FROM Track WHERE Milliseconds > 36000000",                      
])
s = graph.run_agent("List any tracks longer than 10 hours", cross_check=True)
check("primary query still ran fine", s["status"] == "success")
check("no error was ever raised", all(h["stage"] == "success" for h in s["history"]))
check("disagreement detected", s["agreement"] == "disagree", s["agreement"])
check("confidence downgraded", s["confidence"] == "low")
check("both result sets kept", len(s["rows"]) > 0 and len(s["cross_check_rows"]) == 0)

print("\n9. Cross-check confirms a correct answer")
scripted([
    "SELECT COUNT(*) AS n FROM Customer",
    "SELECT COUNT(CustomerId) AS total FROM Customer",                               
])
s = graph.run_agent("How many customers are there?", cross_check=True)
check("agreement detected", s["agreement"] == "agree", s["agreement"])
check("confidence high", s["confidence"] == "high")

print("\n10. Different column shape is not a disagreement")
scripted([
    "SELECT FirstName, LastName FROM Customer WHERE CustomerId = 1",
    "SELECT FirstName || ' ' || LastName AS Name FROM Customer WHERE CustomerId = 1",
])
s = graph.run_agent("Who is customer 1?", cross_check=True)
check("still counts as agreement", s["agreement"] == "agree", s["agreement"])

print("\n10b. ...and the comparison is symmetric (either column order)")
scripted([
    "SELECT FirstName || ' ' || LastName AS Name FROM Customer WHERE CustomerId = 1",
    "SELECT FirstName, LastName FROM Customer WHERE CustomerId = 1",
])
s = graph.run_agent("Who is customer 1?", cross_check=True)
check("agreement in the reverse order too", s["agreement"] == "agree", s["agreement"])

print("\n11. A broken cross-check query is inconclusive, not a disagreement")
scripted([
    "SELECT COUNT(*) AS n FROM Customer",
    "SELECT COUNT(*) FROM NoSuchTable",                                    
])
s = graph.run_agent("How many customers are there?", cross_check=True)
check("marked inconclusive", s["agreement"] == "inconclusive", s["agreement"])
check("primary answer not discarded", s["status"] == "success" and len(s["rows"]) == 1)

print("\n12. Cross-check is off by default (it costs an extra call)")
scripted(["SELECT COUNT(*) AS n FROM Customer"])
s = graph.run_agent("How many customers are there?")
check("skipped", s["agreement"] == "skipped")
check("confidence unverified", s["confidence"] == "unverified")

print("\n13. Grading logic")
check("column split still counts as correct",
      compare([{"F": "Helena", "L": "Holy"}], [{"Name": "Helena Holy"}])[1])
check("wrong number is not correct", not compare([{"n": 59}], [{"n": 58}])[1])
check("empty == empty", compare([], [])[0])

print(f"\n{'ALL OFFLINE TESTS PASSED' if not failures else str(failures) + ' TEST(S) FAILED'}\n")
raise SystemExit(1 if failures else 0)
