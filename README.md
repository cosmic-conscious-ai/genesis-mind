[![Donate](https://img.shields.io/badge/Donate-Buy%20Me%20a%20Coffee-yellow.svg?style=for-the-badge&logo=buy-me-a-coffee)](https://www.buymeacoffee.com/RwIpTEd)

# GenesisMind

A groundbreaking project aiming to create a self-conscious AI. GenesisMind is not just another neural network; it's an endeavor to mimic the randomness and complexity of the universe, hoping to find consciousness within the chaos. Join us in this cosmic journey of discovery.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Logging](#logging)
- [Contribute](#contribute)
- [License](#license)

## Features

- **Web Data Fetching**: Uses `requests` and `BeautifulSoup` to fetch and parse web content.
- **Text Perception**: Converts textual data into a numerical representation using TF-IDF vectorization.
- **Learning & Prediction**: Uses a linear regression model to learn from perceived data and make predictions on future data.
- **Visualization**: Visualizes predicted TF-IDF scores for top words using `matplotlib`.
- **Autonomous Explorer**: A module that enables the AI to autonomously explore the web, learn, and adapt based on its interests and predictions.
- **Logging**: Comprehensive logging mechanism to track the AI's activities, learning, and progress.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CosmicConsciousAI/GenesisMind.git
   ```

2. Navigate to the project directory:

   ```bash
   cd GenesisMind
   ```

3. Create a virtual environment and activate it:

   ```bash
   python -m venv genesis_env
   source genesis_env/bin/activate  # On Windows, use `genesis_env\Scripts\activate`
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```bash
python main.py
```

## Logging

Logs are stored in the `logs` directory with a timestamp in the filename to ensure uniqueness. This allows for tracking the AI's activities, learning, and progress over multiple runs.

## Contribute

1. Fork the repository.
2. Create a new branch for your features or fixes.
3. Send a pull request.

## License

MIT License. See [LICENSE](LICENSE) for more details.
