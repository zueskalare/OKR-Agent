"""
okr_demo.py

Demonstration of storing OKR tasks and related messages in SQLiteRAGMemory.
This script:
  - Creates a task, subtask, and key result
  - Saves conversational messages linked to the OKR entries
  - Performs a similarity-based memory retrieval of messages
  - Prints out the contents of the OKR tables and retrieved history
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.getcwd())

from AtomAgents.memory.sqlite_rag_memory import SQLiteRAGMemory


def main():
    # Initialize memory (in a local SQLite file)
    db_path = "okr_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    # Ensure OpenAI API key is set for GPT embeddings
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to use GPT embeddings.")
        sys.exit(1)
    # Initialize memory; uses GPTEmbedding internally for embeddings
    mem = SQLiteRAGMemory(sqlite_path=db_path, k=3)

    # Create an OKR: Task -> Subtask
    task_id = mem.create_task(
        job_id="job1", description="Increase revenue", status="submitted"
    )
    subtask_id = mem.create_subtask(
        task_id, description="Launch new feature", status="pending"
    )
    print(f"Created Task(id={task_id}), Subtask(id={subtask_id})")
    # Create a Key Result under the subtask
    key_result_id = mem.create_key_result(
        task_id=task_id,
        description="Feature adoption > 20%",
        subtask_id=subtask_id,
        achieved=False,
    )
    print(f"Created KeyResult(id={key_result_id})")

    # Save messages linked to the key result when the subtask is completed
    mem.update_subtask_status(subtask_id, status="done")
    mem.save_context(
        inputs={"input": "Development is complete"},
        outputs={"output": "Feature deployed to production"},
        job_id="job1",
        task_id=task_id,
        key_result_id=key_result_id,
    )
    mem.save_context(
        inputs={"input": "Users report high satisfaction"},
        outputs={"output": "Adoption rate reached 25%"},
        job_id="job1",
        task_id=task_id,
        key_result_id=key_result_id,
    )

    # Perform a memory retrieval
    query = "Tell me about the feature adoption"
    history = mem.load_memory_variables({"input": query}).get("history", [])
    print(f"\nRetrieved messages for query '{query}':")
    for idx, msg in enumerate(history, 1):
        print(f"  {idx}. {msg}")

    # Display the OKR tables directly via SQL
    print("\n=== Tasks Table ===")
    for row in mem.conn.execute("SELECT id, job_id, description, status FROM tasks"):
        print(row)

    print("\n=== Subtasks Table ===")
    for row in mem.conn.execute(
        "SELECT id, task_id, description, status FROM subtasks"
    ):
        print(row)

    # No key_results table in this OKR architecture


if __name__ == "__main__":
    main()
