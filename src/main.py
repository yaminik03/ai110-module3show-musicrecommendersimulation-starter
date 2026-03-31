from typing import Dict

try:
    # when running as a script: python src/main.py
    from recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    # when running as a module: python -m src.main
    from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print("Loaded songs:", len(songs))

    # Define multiple example user profiles for testing
    profiles = [
        (
            "High-Energy Pop",
            {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 0.9,
                "likes_acoustic": False,
            },
        ),
        (
            "Chill Lofi",
            {
                "favorite_genre": "lofi",
                "favorite_mood": "chill",
                "target_energy": 0.35,
                "likes_acoustic": True,
            },
        ),
        (
            "Intense / Conflicted",
            {
                "favorite_genre": "rock",
                "favorite_mood": "sad",
                "target_energy": 0.95,
                "likes_acoustic": False,
            },
        ),
    ]

    sep = "-" * 40

    # Helper to print recommendations for a profile
    def print_recs(profile_name: str, prefs: Dict, k: int = 5) -> None:
        print(f"\n=== Profile: {profile_name} ===\n")

        recs = recommend_songs(prefs, songs, k=k)

        for song, score, explanation in recs:
            title = song.get("title") or song.get("name") or "Unknown Title"
            artist = song.get("artist") or "Unknown Artist"

            print(sep)
            print(f"🎵  {title.upper()}")
            print(f"   by: {artist}")
            print(f"   Score:   {score:.2f}%")
            print(f"   Explanation: {explanation}")
            print(sep)
            print()

    # Run all profiles
    for name, prefs in profiles:
        print_recs(name, prefs, k=5)


if __name__ == "__main__":
    main()