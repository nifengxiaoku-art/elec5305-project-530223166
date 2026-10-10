from msclap import CLAP


PROMPTS = [
    "sound of a dog barking",
    "sound of a rooster",
    "sound of a pig",
    "sound of a cow",
    "sound of a frog",
    "sound of a cat",
    "sound of a hen",
    "sound of flying insects",
    "sound of a sheep",
    "sound of a crow",
]


def main():
    print("Loading Microsoft CLAP...")
    model = CLAP()
    print("CLAP loaded successfully.")

    embeddings = model.get_text_embeddings(PROMPTS)
    print(f"Text embeddings shape: {tuple(embeddings.shape)}")

    print("\nPrompt embedding preview:")
    for prompt, embedding in zip(PROMPTS, embeddings):
        print(f"- {prompt}: norm={float(embedding.norm()):.4f}")


if __name__ == "__main__":
    main()
