from genesis_mind import GenesisMind
import logging


def main(logger):
    NUM_CLASSES = 1000
    mind = GenesisMind(NUM_CLASSES)

    # Start with a generic search term to fetch initial data
    initial_data = mind.explorer.autonomous_explore(
        "https://www.bing.com/search?q=Artificial+Intelligence", max_iterations=1)

    if not initial_data:
        logger.error("No initial data fetched. Exiting.")
        return

    # Train the model on the fetched data
    text_data = [item["data"]
                 for item in initial_data if item["data_type"] == "text"]
    if text_data:
        mind.text_model.fit_vectorizer(text_data)
        mind.text_model.train()

    # Now, perceive each data point and use recursive learning
    for data_item in initial_data:
        mind.perceive(data_item["data"], data_item["data_type"])
        mind.recursive_learn(data_item["data"], data_item["data_type"])
        mind.train(data_item["data_type"])

    if not mind.recall():
        logger.error(
            "GenesisMind's memory is empty. Please provide some initial data before starting autonomous exploration.")
        return

    # Predict using the last data (assuming it's text for demonstration purposes)
    prediction = mind.predict(initial_data[-1]["data"])
    if not prediction:
        logger.warning("Failed to make a prediction.")
        return

    logger.info(f"GenesisMind's prediction for the next state: {prediction}")

    # For demonstration purposes, let's evaluate and adapt using the last two data points
    if len(initial_data) >= 2:
        eval_score = mind.evaluate(
            initial_data[-2]["data"], initial_data[-1]["data"], initial_data[-1]["data_type"])
        logger.info(f"Evaluation Score: {eval_score}")
        mind.adapt(eval_score)
    else:
        logger.warning("Not enough data points for evaluation.")

    # Autonomous exploration and learning
    logger.info("Starting autonomous exploration...")
    mind.explorer.continuous_learning(max_iterations=5)

    # Plot the training curve to visualize the AI's performance over time
    mind.plot_training_curve()
    logger.info("Training curve plotted.")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    try:
        main(logger)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
