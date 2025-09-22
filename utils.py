#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import torch
import numpy as np
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition


# In[ ]:


# --------------------------
# CONFIGURATION
# --------------------------
DB_FILE = "speaker_db.json"
THRESHOLD = 0.60  # similarity threshold (0–1). Adjust based on your data.

# --------------------------
# LOAD MODEL
# --------------------------
print("[INFO] Loading ECAPA-TDNN model...")
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# --------------------------
# LOAD OR INIT DATABASE
# --------------------------
def init_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            speaker_db = json.load(f)
        print(f"[INFO] Loaded {len(speaker_db)} speakers from {DB_FILE}")
    else:
        speaker_db = {}
        print(f"[INFO] Starting with empty database")
    return speaker_db

speaker_db = init_database()

# --------------------------
# UTILS
# --------------------------
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def extract_embedding(file_path):
    """Extract ECAPA-TDNN speaker embedding from audio file."""
    signal, fs = sf.read(file_path)
    if signal.ndim > 1:  # stereo to mono
        signal = np.mean(signal, axis=1)
    tensor_signal = torch.tensor(signal).unsqueeze(0)
    embedding = verification.encode_batch(tensor_signal)
    return embedding.squeeze().detach().cpu().numpy()

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(speaker_db, f)

# --------------------------
# MAIN FUNCTION
# --------------------------
def process_speaker(file_path, mode="add", threshold=THRESHOLD):
    """
    mode="add": fuzzy match + add to DB (default)
    mode="check": fuzzy match only (dry-run)
    """
    global speaker_db
    
    # Reload database in case it was modified by another process
    speaker_db = init_database()
    
    embedding = extract_embedding(file_path)
    emb_list = embedding.tolist()

    matches = []
    for spk_id, emb_json_list in speaker_db.items():
        embeddings = [np.array(json.loads(e)) for e in emb_json_list]
        sims = [cosine_similarity(embedding, emb) for emb in embeddings]
        best_sim = max(sims)
        if best_sim >= threshold:
            matches.append((spk_id, best_sim))

    result = {
        "matches": [],
        "message": "",
        "new_speaker": None
    }
    
    if matches:
        for spk, sim in sorted(matches, key=lambda x: x[1], reverse=True):
            result["matches"].append({"speaker": spk, "similarity": f"{sim:.2f}"})
    
    if mode == "add":
        if matches:
            # Append to best match
            top_speaker, top_sim = max(matches, key=lambda x: x[1])
            speaker_db[top_speaker].append(json.dumps(emb_list))
            result["message"] = f"Appended to {top_speaker} (similarity: {top_sim:.2f})"
        else:
            # New speaker
            new_id = f"speaker_{len(speaker_db)+1}"
            speaker_db[new_id] = [json.dumps(emb_list)]
            result["message"] = f"New speaker added: {new_id}"
            result["new_speaker"] = new_id

        save_db()
    else:
        if matches:
            result["message"] = "Matches found but not added to database (check mode)"
        else:
            result["message"] = "No matches found above threshold"
    
    return result

def get_speakers_list():
    """Return list of all speakers in database"""
    return list(speaker_db.keys())

# --------------------------
# COMMAND LINE USAGE
# --------------------------
if __name__ == "__main__":
    print("\n[INFO] Unified Speaker Tool Ready.")
    print("Usage:")
    print("  python utils.py add path/to/audio.wav   # fuzzy match + add to DB")
    print("  python utils.py check path/to/audio.wav # fuzzy match only (dry-run)\n")

    import sys
    if len(sys.argv) == 3:
        mode = sys.argv[1].lower()
        file_path = sys.argv[2]
        if mode not in ["add", "check"]:
            print("[ERROR] Mode must be 'add' or 'check'")
        elif not os.path.exists(file_path):
            print("[ERROR] File does not exist.")
        else:
            result = process_speaker(file_path, mode=mode)
            print(result["message"])
            if result["matches"]:
                print("Matches found:")
                for match in result["matches"]:
                    print(f"  - {match['speaker']}: similarity {match['similarity']}")
    else:
        print("[ERROR] Invalid arguments.")

