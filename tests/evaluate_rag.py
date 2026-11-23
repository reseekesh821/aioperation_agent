import json
from pathlib import Path
from backend.app.agents.agent_graph import agent_graph

file_path = Path(__file__).with_name("rag_cases.json")
cases = json.loads(file_path.read_text(encoding="utf-8"))

for case in cases:
    passed_count = 0

    for attempt in range(3):
        result = agent_graph.invoke({
            "messages": [("user", case["question"])]
        })

        answer = result["messages"][-1].content

        passed = all(
            term.lower() in answer.lower()
            for term in case["required_terms"]
        )

        if passed:
            passed_count += 1

        print(f"Run {attempt + 1}: {'PASS' if passed else 'FAIL'}")
        print(f"Answer: {answer}")

    print(f"Pass rate: {passed_count}/3\n")