"""
sqlite_memory_demo.py

Demonstrates SQLiteRAGMemory with multiple tasks, subtasks, key results, and messages.
Creates 3 tasks with 5 subtasks, adds 20 messages, and searches for messages
related to a new query.
"""
import os
import sys

# project root on path
sys.path.insert(0, os.getcwd())

from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Export it to run this demo.")
        sys.exit(1)

    db_path = "sqlite_memory_demo.db"
    # Remove existing demo DB
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize memory (will use GPTEmbedding by default)
    mem = SQLiteRAGMemory(sqlite_path=db_path, k=5)

    # Create 3 tasks
    tasks = [
        ("job1", "Sports Updates"),
        ("job2", "Finance Updates"),
        ("job3", "Tech News"),
    ]
    task_ids = {}
    for job_id, desc in tasks:
        tid = mem.create_task(job_id=job_id, description=desc)
        task_ids[job_id] = tid

    # Create 5 subtasks across tasks
    subtask_defs = [
        ("job1", task_ids["job1"], "Football Matches"),
        ("job1", task_ids["job1"], "Basketball Games"),
        ("job2", task_ids["job2"], "Stock Market"),
        ("job2", task_ids["job2"], "Cryptocurrency"),
        ("job3", task_ids["job3"], "AI Developments"),
    ]
    # Map subtask description to (job_id, task_id, subtask_id)
    subtask_map = {}
    for job_id, tid, desc in subtask_defs:
        sid = mem.create_subtask(task_id=tid, description=desc, status="in_progress")
        subtask_map[desc] = (job_id, tid, sid)

    # Create one key result per subtask
    kr_ids = {}
    for desc, (job_id, tid, sid) in subtask_map.items():
        kr = mem.create_key_result(
            task_id=tid,
            description=f"KR for {desc}",
            subtask_id=sid,
            achieved=False,
        )
        kr_ids[desc] = kr

    # Define 4 messages per subtask (total 20)
    messages = {
        "Football Matches": [
            "Last night, Team A beat Team B 3-1.",
            "Player X scored a hat-trick.",
            "The stadium was sold out.",
            "Excellent defensive performance by Team A.",
        ],
        "Basketball Games": [
            "The Lakers won their 10th game in a row.",
            "LeBron scored 30 points.",
            "An impressive three-pointer clinched the game.",
            "Season MVP candidates are lined up.",
        ],
        "Stock Market": [
            "The S&P 500 rose by 2% today.",
            "Tech stocks led the rally.",
            "Interest rate concerns persist.",
            "Analysts predict further growth.",
        ],
        "Cryptocurrency": [
            "Bitcoin surpassed $50k again.",
            "Ethereum gas fees have dropped.",
            "Regulatory news affected the market.",
            "New DeFi protocol launched.",
        ],
        "AI Developments": [
            "A new GPT model has been released.",
            "Researchers achieved 99% accuracy on benchmarks.",
            "Open-source libraries are improving rapidly.",
            "AI ethics discussions are intensifying.",
        ],
    }

    # Save messages into memory
    for desc, msgs in messages.items():
        job_id, tid, sid = subtask_map[desc]
        kr = kr_ids[desc]
        # mark subtask done to simulate completion
        mem.update_subtask_status(sid, status="done")
        for m in msgs:
            mem.save_context(
                inputs={"text": m}, outputs={},
                job_id=job_id, task_id=tid, key_result_id=kr
            )

    # New query to search
    query = "Tell me about the football game performance"
    result = mem.load_memory_variables({"input": query})
    history = result.get("history", [])

    print(f"Query: {query}\nRetrieved top messages:")
    for i, h in enumerate(history, 1):
        print(f"  {i}. {h}")

if __name__ == "__main__":
    main()
