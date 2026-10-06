from __future__ import annotations

from collections import Counter, deque
import time


class PlayerMemory:
    def __init__(self, max_history=15, max_age=180):
        self.players = {}
        self.max_history = max_history
        self.max_age = max_age

    def observe(self, track_id, jersey=None, jersey_confidence=0.0,
                bbox=None, detection_confidence=0.0):
        now = time.time()
        track_id = int(track_id)

        player = self.players.setdefault(
            track_id,
            {
                "track_id": track_id,
                "jersey": None,
                "jersey_confidence": 0.0,
                "jersey_votes": deque(maxlen=self.max_history),
                "bbox": None,
                "detection_confidence": 0.0,
                "last_seen": now,
            },
        )

        player["last_seen"] = now
        player["bbox"] = bbox
        player["detection_confidence"] = float(detection_confidence)

        if jersey is not None and jersey_confidence > 0:
            player["jersey_votes"].append(
                (str(jersey), float(jersey_confidence))
            )

            scores = Counter()
            for number, confidence in player["jersey_votes"]:
                scores[number] += confidence

            best_number, score = scores.most_common(1)[0]
            player["jersey"] = best_number
            player["jersey_confidence"] = min(
                0.99, score / max(1.0, len(player["jersey_votes"]))
            )

        return player

    def get(self, track_id):
        return self.players.get(int(track_id))

    def prune(self):
        now = time.time()
        stale = [
            track_id
            for track_id, player in self.players.items()
            if now - player["last_seen"] > self.max_age
        ]
        for track_id in stale:
            del self.players[track_id]

    def snapshot(self):
        return {
            str(track_id): {
                "track_id": player["track_id"],
                "jersey": player["jersey"],
                "jersey_confidence": round(player["jersey_confidence"], 3),
                "last_seen": player["last_seen"],
            }
            for track_id, player in self.players.items()
        }
