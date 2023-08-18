from genesis_mind import GenesisMind
import logging
import atexit
from logger import setup_logger


def main(logger):
    ENDPOINT = "https://www.duckduckgo.com/html?q="
    NUM_CLASSES = 1000
    mind = GenesisMind(NUM_CLASSES)

    # Load the saved models and state
    mind.load_models()
    mind.load_state()
    mind.explorer.load_state()

    # Register the save functions to be called on termination
    atexit.register(mind.save_models)
    atexit.register(mind.save_state)
    atexit.register(mind.explorer.save_state)

    if not mind.is_state_loaded():
        logger.info("This is the first run or no state has been saved yet.")
        # Start with a generic search term to fetch initial data
        initial_data = mind.explorer.autonomous_explore(
            f"{ENDPOINT}Artificial+Intelligence", ENDPOINT)

        if not initial_data:
            logger.error("No initial data fetched. Exiting.")
            return
            
        # Combine all the fetched texts
        combined_text = ' '.join([item["data"]
                                  for item in initial_data if item["data_type"] == "text"])

        # Train the model on the combined text
        if combined_text:
            mind.text_model.fit_vectorizer([combined_text])
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

        # Predict a new search term using the combined text
        prediction = mind.predict(combined_text)
        if not prediction:
            logger.warning("Failed to make a prediction.")
            return

        logger.info(
            f"GenesisMind's prediction for the next state: {prediction}")

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
    mind.explorer.continuous_learning(ENDPOINT)

    # Plot the training curve to visualize the AI's performance over time
    mind.plot_training_curve()
    logger.info("Training curve plotted.")


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    try:
        main(logger)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
