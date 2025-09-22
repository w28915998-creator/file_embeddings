import os
import json
import torch
import numpy as np
import torchaudio
from torchaudio.transforms import Vad
from speechbrain.pretrained import SpeakerRecognition




# --------------------------
# CONFIGURATION
# --------------------------
DB_FILE = "speaker_db.json"
THRESHOLD = 0.80  # similarity threshold (0–1)

TARGET_SR = 16000  # ECAPA model expects 16kHz
MIN_CLIP_SEC = 1.5  # discard too short clips (in seconds)

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

def preprocess_audio(file_path, target_sr=TARGET_SR):
    """Load audio, resample, normalize, remove silence."""
    waveform, sr = torchaudio.load(file_path)

    # Resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize amplitude
    waveform = waveform / (waveform.abs().max() + 1e-9)

    # Voice Activity Detection (remove silence)
    vad = Vad(sample_rate=target_sr)
    waveform = vad(waveform)

    # Check duration
    if waveform.shape[1] < target_sr * MIN_CLIP_SEC:
        print("[WARN] Clip too short after VAD, may reduce accuracy.")

    return waveform

def extract_embedding(file_path):
    """Extract ECAPA-TDNN speaker embedding from audio file with preprocessing."""
    tensor_signal = preprocess_audio(file_path)
    embedding = verification.encode_batch(tensor_signal)
    return embedding.squeeze().detach().cpu().numpy()

def get_speaker_centroid(emb_json_list):
    """Average all stored embeddings for a speaker."""
    embeddings = [np.array(json.loads(e)) for e in emb_json_list]
    return np.mean(embeddings, axis=0)

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
        centroid = get_speaker_centroid(emb_json_list)
        sim = cosine_similarity(embedding, centroid)
        if sim >= threshold:
            matches.append((spk_id, sim))

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
            # Append new embedding to the best matching speaker
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
