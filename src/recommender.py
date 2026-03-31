from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs for the given user profile."""
        scored = []

        for song in self.songs:
            score = self._score(song, user)
            scored.append((song, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return short human-readable reasons why a song matches a user."""
        reasons = []

        if song.genre == user.favorite_genre:
            reasons.append("matches your favorite genre")

        if song.mood == user.favorite_mood:
            reasons.append("matches your mood")

        if abs(song.energy - user.target_energy) < 0.2:
            reasons.append("has similar energy")

        if user.likes_acoustic and song.acousticness > 0.5:
            reasons.append("fits your acoustic preference")

        return ", ".join(reasons) if reasons else "generally a good match"

    def _score(self, song: Song, user: UserProfile) -> float:
        """Compute a normalized similarity score for a song and user profile."""
        from math import exp

        def gaussian_sim(x: float, mu: float, sigma: float = 0.10) -> float:
            """Return Gaussian similarity between x and mu (0..1)."""
            # guard against None
            if x is None or mu is None:
                return 0.0
            d2 = (x - mu) ** 2
            return float(exp(-d2 / (2 * sigma * sigma)))

        # categorical sims (simple exact-match; can be replaced with matrix)
        sim_genre = 1.0 if song.genre == user.favorite_genre else 0.0
        sim_mood = 1.0 if song.mood == user.favorite_mood else 0.0

        # Experimental weight shift: halve genre importance, double energy importance
        # - genre original max 2.0 -> halved to 1.0
        # - energy original max 1.0 -> doubled to 2.0
        # Mood remains at 1.0. Total max remains 4.0 so normalization is unchanged.
        pts_genre = (2.0 * 0.5) * sim_genre
        pts_mood = 1.0 * sim_mood
        pts_energy = (1.0 * 2.0) * gaussian_sim(song.energy, user.target_energy, sigma=0.10)

        total_points = pts_genre + pts_mood + pts_energy
        # max possible points = 4.0
        score = total_points / 4.0
        return float(score)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV or JS-style file and return a list of dicts."""
    import csv
    import re
    songs: List[Dict] = []

    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    def parse_int(v: Optional[str]) -> Optional[int]:
        """Parse a string as an int, returning None on empty or invalid input."""
        if v is None or v.strip() == "":
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    def parse_float(v: Optional[str]) -> Optional[float]:
        """Parse a string as a float, returning None on empty or invalid input."""
        if v is None or v.strip() == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    # Try CSV first, but fall back to parsing a JS-style `songs` array (some starter data
    # is stored as a JS file). Detect JS by looking for the `const songs` marker.
    print(f"Loading songs from {csv_path}...")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        text = fh.read()

    # If file looks like a JS module with `const songs = [ ... ]`, parse that format
    if "const songs" in text and "title:" in text:
        # find all `{ ... }` blocks that represent song objects
        obj_blocks = re.findall(r"\{([^}]*)\}", text, flags=re.DOTALL)
        for blk in obj_blocks:
            # find key:value pairs inside the block
            pairs = re.findall(r"(\w+)\s*:\s*(?:\"([^\"]*)\"|'([^']*)'|([0-9]+(?:\.[0-9]+)?))", blk)
            if not pairs:
                continue
            song: Dict[str, object] = {}
            for key, v1, v2, v3 in pairs:
                raw = None
                if v1:
                    raw = v1
                elif v2:
                    raw = v2
                elif v3:
                    raw = v3

                if key in int_fields:
                    song[key] = parse_int(raw)
                elif key in float_fields:
                    song[key] = parse_float(raw)
                else:
                    song[key] = raw if raw is not None else ""

            # only add if id exists
            if song.get("id") is not None:
                songs.append(song)

        return songs

    # Otherwise try CSV parsing
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row:
                continue

            clean: Dict[str, Optional[object]] = {}
            for k, v in row.items():
                # normalize header keys by stripping whitespace
                key = k.strip() if isinstance(k, str) else k
                if v is None:
                    clean[key] = None
                elif isinstance(v, str):
                    clean[key] = v.strip()
                else:
                    clean[key] = str(v).strip()

            song: Dict[str, object] = {}
            for k, v in clean.items():
                if k in int_fields:
                    song[k] = parse_int(v if isinstance(v, str) else None)
                elif k in float_fields:
                    song[k] = parse_float(v if isinstance(v, str) else None)
                else:
                    song[k] = v if v is not None else ""

            songs.append(song)

    return songs


def normalize_tempo(t: float) -> float:
    return (t - 55) / (180 - 55)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank songs by similarity to user_prefs and return top-k results.

    Uses score_song(song, user_prefs) to compute a normalized score (0..1)
    and a list of human-readable reasons. Returns a list of tuples
    (song_dict, score_pct, explanation_str) sorted by score descending.
    """
    # Compute scores for each song (pythonic, using a list comprehension)
    scored = [
        (
            song,
            round(score * 100, 2),
            ", ".join(reasons) if reasons else "similar overall profile",
        )
        for song, (score, reasons) in (
            (song, score_song(song, user_prefs)) for song in songs
        )
    ]

    # Sort by score (second element) descending and return top-k
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def score_song(song: Dict, user_prefs: Dict, sigma: float = 0.10) -> Tuple[float, List[str]]:
    """
    Score a single song (dict) against user_prefs (dict) using the algorithm recipe:
    - Genre match: up to +2.0 points (exact match -> +2.0)
    - Mood match: up to +1.0 point (exact match -> +1.0)
    - Energy proximity: up to +1.0 point using a Gaussian similarity (peak=1.0 at exact match)

    Returns (normalized_score, reasons) where normalized_score is in [0,1] (total_points / 4.0)
    and reasons is a list of human-readable strings like "genre match (+2.0)".
    """
    from math import exp

    def gaussian_sim(x: Optional[float], mu: Optional[float], sigma: float) -> float:
        if x is None or mu is None:
            return 0.0
        try:
            d2 = (float(x) - float(mu)) ** 2
        except (ValueError, TypeError):
            return 0.0
        return float(exp(-d2 / (2 * sigma * sigma)))

    reasons: List[str] = []

    # flexible key lookup: accept either 'favorite_genre' or 'genre' in user_prefs
    user_genre = user_prefs.get("favorite_genre") or user_prefs.get("genre")
    user_mood = user_prefs.get("favorite_mood") or user_prefs.get("mood")
    user_energy = user_prefs.get("target_energy") or user_prefs.get("energy")

    total_points = 0.0

    # Genre (halved importance for experiment)
    song_genre = song.get("genre")
    if song_genre and user_genre and song_genre == user_genre:
        pts = 2.0 * 0.5  # halved
        total_points += pts
        reasons.append(f"genre match (+{pts:.1f})")

    # Mood
    song_mood = song.get("mood")
    if song_mood and user_mood and song_mood == user_mood:
        pts = 1.0
        total_points += pts
        reasons.append(f"mood match (+{pts:.1f})")

    # Energy proximity (numeric) - doubled importance for experiment
    song_energy = song.get("energy")
    sim_e = gaussian_sim(song_energy, user_energy, sigma)
    pts_e = 1.0 * 2.0 * sim_e
    if pts_e > 0:
        reasons.append(f"energy proximity (+{pts_e:.2f})")
    total_points += pts_e

    # Final normalization: max points = 4.0
    score_norm = float(total_points / 4.0)
    return score_norm, reasons