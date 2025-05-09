import random
from predict import Predictor
from prepare_data import load_flickr30k_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm


def compute_rouge_l_score(predictor, dataset, sampled_indices: list[int]):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for idx in tqdm(sampled_indices, desc="Evaluating ROUGE-L"):
        data = dataset[idx]
        image = data["image"]
        references = data["caption"]  # list of reference captions

        generated_caption = predictor.predict(
            image=image, prompt="")[0].strip()

        # Compute ROUGE-L score with the best of the reference captions
        best_score = max(
            [scorer.score(ref, generated_caption)[
                "rougeL"].fmeasure for ref in references]
        )

        scores.append(best_score)

    average_rouge_l = sum(scores) / len(scores)
    print(
        f"\nAverage ROUGE-L Score samples: {average_rouge_l:.4f}")
    return average_rouge_l

# Usage


def main():
    predictor = Predictor()
    predictor.setup()

    dataset = load_flickr30k_dataset()["test"]
    sample_size = 1000
    sampled_indices = random.sample(range(len(dataset)), sample_size)
    rouge_scores = []
    for model_id in range(0, 12):  # Loop through 11 models
        print(f"\nEvaluating Model {model_id}")
        # Assuming a method to load specific models
        predictor.load_model(f"output/output-{model_id:03d}.pt")
        rouge_score = compute_rouge_l_score(
            predictor, dataset, sampled_indices)
        rouge_scores.append(rouge_score)
    print("\nROUGE-L Scores for all models:")
    for model_id, score in enumerate(rouge_scores):
        print(f"Model {model_id}: {score:.4f}")


if __name__ == "__main__":
    main()
