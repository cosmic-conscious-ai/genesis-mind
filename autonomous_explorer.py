from bs4 import BeautifulSoup
from data_processing import DataProcessor
import random
import re
import logging
import pickle
import os


class AutonomousExplorer:
    """
    A class that enables the GenesisMind to autonomously explore the web, learn, and adapt.
    """

    def __init__(self, genesis_mind):
        self.genesis_mind = genesis_mind
        self.visited_urls = set()
        self.data_processor = DataProcessor()
        self.logger = logging.getLogger(self.__class__.__name__)

    def discover_links(self, content: str) -> list:
        try:
            if not content:
                self.logger.warning(
                    "Empty content provided. No links discovered.")
                return []

            soup = BeautifulSoup(content, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)
                     if a['href'] and not re.search(r'(^#|javascript:|mailto:)', a['href'])
                     and a['href'].startswith(('http://', 'https://'))]  # Check if the link starts with http:// or https://

            self.logger.debug(f"Discovered {len(links)} links from content.")
            return links
        except Exception as e:
            self.logger.error(f"Error discovering links: {e}")
            return []

    def autonomous_explore(self, search_url: str):
        """
        Start with a search URL and explore the web autonomously.
        """
        if not search_url:
            self.logger.error("Invalid search URL provided. Exiting.")
            return []

        fetched_data = []

        try:
            current_url = search_url

            while current_url and current_url not in self.visited_urls:
                content = self.data_processor.fetch_web_data(current_url)
                soup = BeautifulSoup(content, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True)

                if content:
                    # Assuming all fetched data is of type "text" for now
                    fetched_data.append(
                        {"data": text_content, "data_type": "text"})
                    self.genesis_mind.recursive_learn(content)
                    self.visited_urls.add(current_url)
                    links = self.discover_links(content)

                    # Instead of choosing a random link, we'll iterate over all links
                    for link in links:
                        if link not in self.visited_urls:
                            current_url = link
                            break
                    else:
                        # If all links are visited or no links are found, evolve the search query
                        current_url = self.evolve_search_query()
                else:
                    self.logger.error(
                        f"Failed to fetch content from {current_url}. Trying a different search term.")
                    current_url = self.evolve_search_query()
        except Exception as e:
            self.logger.error(f"Error during autonomous exploration: {e}")
            return []

        # Return the fetched data
        return fetched_data

    def evolve_search_query(self):
        """
        Evolve the search query based on the AI's interests, predictions, and introspection.
        """
        introspective_queries = [
            "Principles of Artificial Intelligence",
            "History of AI",
            "Machine Learning Techniques",
            "Neural Networks Basics"
        ]

        # Default to introspective query
        default_query = random.choice(introspective_queries)
        evolved_query = f"https://www.bing.com/search?q={default_query.replace(' ', '+')}"

        try:
            last_memory = self.genesis_mind.recall()

            # Extract data from MemoryItem
            last_memory_data = last_memory.data if last_memory else None

            if not last_memory_data:
                self.logger.warning(
                    "No last memory found. Starting introspection.")
                return evolved_query

            last_prediction = self.genesis_mind.predict(
                last_memory_data)  # Use the extracted data for prediction
            if not last_prediction:
                self.logger.warning(
                    "Model prediction is None. Reverting to introspection.")
                return evolved_query

            # Convert the list of predicted terms to a search query string
            search_query_string = '+'.join(last_prediction)
            evolved_query = f"https://www.bing.com/search?q={search_query_string}"

            self.logger.info(f"Evolved search query to: {search_query_string}")

        except Exception as e:
            self.logger.error(f"Error evolving search query: {e}")
            return evolved_query

        return evolved_query

    def continuous_learning(self):
        """
        Continuously explore, learn, and adapt.
        """
        try:
            while not self.genesis_mind.is_conscious():
                self.logger.info("Starting continuous learning process.")
                search_url = self.evolve_search_query()
                self.autonomous_explore(search_url)
        except Exception as e:
            self.logger.error(f"Error during continuous learning: {e}")

    def save_state(self, path="models/explorer_state.pkl"):
        """
        Save the state of the AutonomousExplorer to disk.
        """
        try:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, 'wb') as f:
                pickle.dump({
                    'visited_urls': self.visited_urls
                }, f)

            self.logger.info(f"AutonomousExplorer state saved to {path}.")
        except Exception as e:
            self.logger.error(f"Error saving AutonomousExplorer state: {e}")

    def load_state(self, path="models/explorer_state.pkl"):
        """
        Load the state of the AutonomousExplorer from disk.
        """
        try:
            if not os.path.exists(path):
                self.logger.warning(f"No saved state found at {path}.")
                return

            with open(path, 'rb') as f:
                saved_state = pickle.load(f)
                self.visited_urls = saved_state['visited_urls']

            self.logger.info(f"AutonomousExplorer state loaded from {path}.")
        except Exception as e:
            self.logger.error(f"Error loading AutonomousExplorer state: {e}")
