from bs4 import BeautifulSoup
from data_processing import DataProcessor
import random
import re
import logging


class AutonomousExplorer:
    """
    A class that enables the GenesisMind to autonomously explore the web, learn, and adapt.
    """

    def __init__(self, genesis_mind):
        self.genesis_mind = genesis_mind
        self.data_processor = DataProcessor()
        self.visited_urls = set()  # Track visited URLs
        self.logger = logging.getLogger(self.__class__.__name__)

    def discover_links(self, content: str) -> list:
        try:
            if not content:
                self.logger.warning(
                    "Empty content provided. No links discovered.")
                return []

            soup = BeautifulSoup(content, 'html.parser')
            links = [a['href'] for a in soup.find_all(
                'a', href=True) if a['href'] and not re.search(r'(^#|javascript:|mailto:)', a['href'])]
            self.logger.debug(f"Discovered {len(links)} links from content.")
            return links
        except Exception as e:
            self.logger.error(f"Error discovering links: {e}")
            return []

    def autonomous_explore(self, search_url: str, max_iterations: int = 10):
        """
        Start with a search URL and explore the web autonomously.
        """
        if not search_url:
            self.logger.error("Invalid search URL provided. Exiting.")
            return []

        fetched_data = []

        try:
            current_url = search_url
            for _ in range(max_iterations):
                if current_url in self.visited_urls:
                    self.logger.info(
                        f"URL {current_url} already visited. Evolving search query.")
                    current_url = self.evolve_search_query()
                    continue

                content = self.data_processor.fetch_web_data(current_url)
                if content:
                    # Assuming all fetched data is of type "text" for now
                    fetched_data.append({"data": content, "data_type": "text"})
                    self.genesis_mind.recursive_learn(content)
                    self.visited_urls.add(current_url)
                    links = self.discover_links(content)
                    current_url = random.choice(
                        [link for link in links if link not in self.visited_urls]) if links else self.evolve_search_query()
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
        evolved_query = f"https://www.google.com/search?q={random.choice(introspective_queries)}"

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

            self.logger.info(f"Evolved search query to: {last_prediction}")

            evolved_query = f"https://www.google.com/search?q={last_prediction}"
            if not evolved_query:
                self.logger.error(
                    "Failed to evolve search query. Reverting to introspection.")
                return evolved_query
        except Exception as e:
            self.logger.error(f"Error evolving search query: {e}")
            return evolved_query

        return evolved_query

    def continuous_learning(self, max_iterations: int = 10):
        """
        Continuously explore, learn, and adapt.
        """
        try:
            self.logger.info("Starting continuous learning process.")
            search_url = self.evolve_search_query()
            self.autonomous_explore(search_url, max_iterations)
        except Exception as e:
            self.logger.error(f"Error during continuous learning: {e}")
