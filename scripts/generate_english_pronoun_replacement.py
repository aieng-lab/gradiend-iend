"""Generate a corrected pronoun candidate set without touching published-cache files."""

from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data


if __name__ == "__main__":
    paths = ensure_english_pronoun_data(
        output_dir="build/english_pronouns_replacement",
        force=True,
    )
    print(paths)
