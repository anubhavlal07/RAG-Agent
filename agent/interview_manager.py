import os
import sys
import json
from time import time
from datetime import datetime
from dotenv import load_dotenv

# Import the general and technical interview functions
from agent.general_agent import run_general_hr_interview, get_session_history as get_hr_history
from agent.technical_agent import interview_loop as run_technical_interview, get_session_history as get_tech_history

# Pinecone lookup
from agent.data_loader import get_candidate_by_phone, load_full_resume_text

load_dotenv()

# ----------------------------------------------------------------
# STEP 1: fetch_and_confirm_candidate
# ----------------------------------------------------------------

def fetch_and_confirm_candidate():
    """
    1. Prompt for a phone number.
    2. Use get_candidate_by_phone() to retrieve metadata.
    3. Display key fields and ask for confirmation.
       If “no,” loop again. If “yes,” return (phone_number, metadata_dict).
    """
    print("\n===== Candidate Lookup =====\n")
    while True:
        phone = input("Please enter your phone number (digits only or formatted): ").strip()
        if not phone:
            continue

        digits = "".join(filter(str.isdigit, phone))
        if len(digits) < 10:
            print("❗ We need at least 10 digits to look up your record. Please try again.")
            continue

        metadata = get_candidate_by_phone(digits)
        if not metadata:
            print(f"❗ No candidate found for phone number: {digits}. Try again or type 'exit'.")
            if phone.lower() in ["exit", "quit"]:
                sys.exit(0)
            continue

        print("\nWe found the following information for you:\n")
        display_fields = {
            "Name": metadata.get("name"),
            "Email": metadata.get("email", "<not provided>"),
            "Education": metadata.get("education", "<not provided>"),
            "Skills": ", ".join(metadata.get("skills", [])) if metadata.get("skills") else "<not provided>",
            "Experience (years)": metadata.get("experience_years", "<not provided>"),
            "Current Role": metadata.get("current_role", "<not provided>")
        }
        for label, val in display_fields.items():
            print(f"  • {label}: {val}")
        print()

        confirm = input("Is this information correct? (yes/no): ").strip().lower()
        if confirm in ["yes", "y"]:
            return digits, metadata
        elif confirm in ["no", "n"]:
            print("Okay, let's try again.\n")
            continue

# ----------------------------------------------------------------
# MAIN FLOW
# ----------------------------------------------------------------

def main():
    try:
        # STEP 1: Lookup & confirm candidate
        phone_number, metadata = fetch_and_confirm_candidate()
        candidate_name = metadata.get("name", "Candidate").replace(" ", "_")

        # STEP 2: Run general HR interview
        hr_session_id = run_general_hr_interview(phone_number, metadata)

        # If candidate declined or exit during general HR, hr_session_id is None
        if hr_session_id is None:
            print("Interview ended during general HR. Goodbye!")
            sys.exit(0)

        # STEP 3: Extract general history
        hr_msgs = get_hr_history(hr_session_id).messages
        general_history = []
        for msg in hr_msgs:
            general_history.append({
                "speaker": msg.type,
                "text": msg.content,
                "timestamp": datetime.now().isoformat()
            })

        # STEP 4: Load full resume text from Pinecone
        resume_text = load_full_resume_text(phone_number)
        if not resume_text:
            print("❗ Could not retrieve full resume. Skipping technical interview.")
            technical_history = []
        else:
            # STEP 5: Run technical interview (seed with general_history)
            tech_session_id = run_technical_interview(resume_text, general_history)

            # STEP 6: Extract technical history, skipping the seeded general messages
            tech_msgs = get_tech_history(tech_session_id).messages
            tech_only = tech_msgs[len(general_history):]
            technical_history = []
            for msg in tech_only:
                technical_history.append({
                    "speaker": msg.type,
                    "text": msg.content,
                    "timestamp": datetime.now().isoformat()
                })

        # STEP 7: Combine and save all in one JSON under agent/conversation/
        combined = general_history + technical_history
        out_dir = os.path.join("agent", "conversations")
        os.makedirs(out_dir, exist_ok=True)

        filename = f"{candidate_name}-{phone_number}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        print(f"\n✅ Interview saved to {filepath}")

    except KeyboardInterrupt:
        print("\nInterview interrupted. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❗ An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
