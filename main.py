# main.py

import sys
from agent.interview_manager import main as run_interview

def main():
    """
    Entry point for the HR + Technical interview flow.
    This will:
      1. Prompt for phone number, confirm candidate details (via Pinecone).
      2. Run the general HR agent.
      3. Run the technical agent (seeded with resume + HR history).
      4. Save a single JSON transcript named {Name}-{Phone}.json.
    """
    try:
        run_interview()
    except KeyboardInterrupt:
        print("\nInterview interrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❗ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
