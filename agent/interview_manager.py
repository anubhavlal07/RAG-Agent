import sys
import json
import datetime
from dotenv import load_dotenv
import os 

from agent.general_agent import fetch_and_confirm_candidate, run_general_hr_interview, get_session_history as get_hr_history
from agent.technical_agent import interview_loop as run_technical_interview, get_session_history as get_tech_history

# We need a helper to fetch the actual resume text from Pinecone
from agent.data_loader import load_full_resume_text

load_dotenv()

def main():
    """
    1. Fetch + confirm candidate from Pinecone → (phone, metadata).
    2. Run general HR agent → hr_session_id.
    3. Fetch and format general history into a list of dicts.
    4. Load full resume text via load_full_resume_text(phone).
    5. Run technical agent with (resume_text, general_history) → tech_session_id.
    6. Fetch and format technical history—BUT skip the first N seeded messages.
    7. Combine both lists and save a single JSON named {Name}-{Phone}.json.
    """
    # Folder containing all resumes to ingest
    CONVERSATION_FOLDER = os.path.join("agent", "conversations")

    try:
        # Ensure the conversation folder exists
        os.makedirs(CONVERSATION_FOLDER, exist_ok=True)

        # STEP 1: Lookup & confirm
        phone_number, metadata = fetch_and_confirm_candidate()
        candidate_name = metadata.get("name", "Unknown").replace(" ", "_")

        # STEP 2: Run the general HR portion → hr_session_id
        hr_session_id = run_general_hr_interview(phone_number, metadata)

        # STEP 3: Retrieve general history messages
        hr_msgs = get_hr_history(hr_session_id).messages
        general_history = []
        for msg in hr_msgs:
            general_history.append({
                "speaker": msg.type,      # e.g. "system" / "human" / "ai"
                "text": msg.content,
                "timestamp": datetime.datetime.now().isoformat()
            })

        # STEP 4: Fetch the candidate's full resume text
        resume_text = load_full_resume_text(phone_number)
        if not resume_text:
            print("❗ Error: could not fetch the full resume text. Technical interview cannot proceed.")
            sys.exit(1)

        # STEP 5: Run the technical agent, passing resume_text + general_history → tech_session_id
        tech_session_id = run_technical_interview(resume_text, general_history)

        # STEP 6: Retrieve technical history messages
        tech_msgs = get_tech_history(tech_session_id).messages

        # We seeded the first len(general_history) messages in the technical history,
        # so skip exactly that many to avoid duplication.
        num_seeded = len(general_history)
        tech_only = tech_msgs[num_seeded:]

        technical_history = []
        for msg in tech_only:
            technical_history.append({
                "speaker": msg.type,
                "text": msg.content,
                "timestamp": datetime.datetime.now().isoformat()
            })

        # STEP 7: Combine and save both general + technical in a single JSON
        combined = general_history + technical_history
        
        # Construct the full path to the file within the conversation folder
        filename = os.path.join(CONVERSATION_FOLDER, f"{candidate_name}-{phone_number}.json")
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        print(f"\n✅ Entire interview saved to {filename}")

    except KeyboardInterrupt:
        print("\nInterview interrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❗ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()