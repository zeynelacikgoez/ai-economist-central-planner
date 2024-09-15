# AI Economist Central Planner

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/zeynelacikgoez/ai-economist-central-planner)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Foundation: An Economic Simulation Framework with Central Planner

Welcome to the **AI Economist Central Planner** repository! This project is an enhanced implementation of the Foundation framework, designed to simulate socio-economic behaviors and dynamics in a society comprising both agents and a central government. Our central planner utilizes advanced machine learning models to optimize economic policies, promoting equality and efficiency within the simulated economy.

### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Using pip](#using-pip)
  - [Installing from Source](#installing-from-source)
- [Getting Started](#getting-started)
- [Simulation](#simulation)
- [Visualization](#visualization)
- [Unit Tests](#unit-tests)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)
- [Contact](#contact)

## Overview

The **AI Economist Central Planner** framework models a society where a central government coordinates economic activities through a central planner. This planner leverages machine learning to develop optimal policies aimed at balancing productivity and equality. By integrating real-world economic data and offering multiple planning strategies, the framework provides a robust platform for studying and developing economic policies using reinforcement learning.

### Key Components

- **Agents:** Represent economic actors (e.g., workers) who follow the directives of the central planner.
- **Central Planner:** Uses advanced ML models to generate and update economic policies based on the current state of the economy.
- **Environment:** Simulates the economic landscape, including resources, production processes, and economic indicators like GDP.
- **Reward Function:** Measures the success of policies based on equality and efficiency metrics.
- **Visualization Tools:** Provide insights into the simulation outcomes through various plots and distributions.

## Features

- **Advanced ML Models:** Utilizes specific neural network architectures with Batch Normalization, Dropout, and Early Stopping to enhance model performance and prevent overfitting.
- **Real Economic Data Integration:** Incorporates real-world GDP data to calibrate and validate the simulation, enhancing realism.
- **Multiple Planning Strategies:** Supports various strategies (e.g., Balance, Equality, Efficiency) allowing comparison and analysis of different policy approaches.
- **Modular Architecture:** Flexible and composable environment structure facilitates easy extensions and modifications.
- **Robust Testing:** Comprehensive unit tests ensure the reliability and correctness of the framework components.
- **Visualization:** Detailed plots for equality, efficiency, and reward metrics to analyze simulation results effectively.

## Installation

### Using pip

To install the AI Economist Central Planner using pip, run:

```bash
pip install ai-economist-central-planner
```

### Installing from Source

1. **Clone this repository to your local machine:**

    ```bash
    git clone https://github.com/zeynelacikgoez/ai-economist-central-planner.git
    cd ai-economist-central-planner
    ```

2. **Create a new virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Für Windows: venv\Scripts\activate
    ```

3. **Install the required dependencies:**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Initialize the database:**

    ```bash
    python -c "from ai_economist.database import init_db; init_db()"
    ```

5. **Set up environment variables:**

    Create a `.env` file in the root directory and add the following:

    ```env
    API_TOKEN=securetoken123
    DATABASE_URL=sqlite:///economy.db
    GDP_DATA_PATH=ai_economist/data/gdp_data.csv
    ```

## Getting Started

To familiarize yourself with the AI Economist Central Planner, explore the tutorials provided in the `tutorials` folder. These notebooks demonstrate how to interact with the simulation, visualize results, and implement various economic policies.

### Multi-Agent Simulations

- [economic_simulation_basic](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/economic_simulation_basic.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/economic_simulation_basic.ipynb)!): Introduction to interacting with and visualizing the simulation.
- [economic_simulation_advanced](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/economic_simulation_advanced.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/economic_simulation_advanced.ipynb)!): Detailed exploration of composable and flexible building blocks within the simulation.
- [optimal_taxation_theory_and_simulation](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/optimal_taxation_theory_and_simulation.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/optimal_taxation_theory_and_simulation.ipynb)!): Demonstrates the use of economic simulations to study optimal taxation policies.
- [covid19_and_economic_simulation](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/covid19_and_economic_simulation.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/covid19_and_economic_simulation.ipynb)!): Introduces a simulation on the COVID-19 pandemic and economy to study health and economic policies.

### Multi-Agent Training

- [multi_agent_gpu_training_with_warp_drive](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb)!): Introduction to our multi-agent reinforcement learning framework [WarpDrive](https://arxiv.org/abs/2108.13976), used to train the COVID-19 and economic simulation.
- [multi_agent_training_with_rllib](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/multi_agent_training_with_rllib.ipynb) ([Try this on Colab](https://colab.research.google.com/github/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/multi_agent_training_with_rllib.ipynb)!): Demonstrates distributed multi-agent reinforcement learning with [RLlib](https://docs.ray.io/en/latest/rllib/index.html).
- [two_level_curriculum_training_with_rllib](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/tutorials/two_level_curriculum_learning_with_rllib.md): Describes implementing two-level curriculum training with [RLlib](https://docs.ray.io/en/latest/rllib/index.html).

To run these notebooks locally, you need [Jupyter](https://jupyter.org). See [Jupyter Installation Guide](https://jupyter.readthedocs.io/en/latest/install.html) for installation instructions and [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/) for examples.

## Simulation

The simulation runs multiple episodes, each consisting of several timesteps. For each strategy (Balance, Equality, Efficiency), the simulation collects key performance indicators (KPIs) such as equality, efficiency, and average rewards. These KPIs are then visualized to analyze the impact of different planning strategies.

### Running the Simulation

1. **Activate the virtual environment:**

    ```bash
    source venv/bin/activate  # Für Windows: venv\Scripts\activate
    ```

2. **Run the main simulation script:**

    ```bash
    python ai_economist/main_simulation.py
    ```

    This script executes simulations for each strategy and generates visualizations of the results.

## Visualization

The framework includes visualization tools to analyze the simulation outcomes effectively. After running the simulation, you will obtain plots showcasing the development of equality, efficiency, and rewards over the episodes for each strategy.

### Types of Visualizations

- **Gleichheitsentwicklung:** Tracks the evolution of equality within the economy.
- **Effizienzentwicklung:** Monitors the changes in economic efficiency.
- **Belohnungsentwicklung:** Observes the progression of average rewards among agents.

Additionally, distributions of income and production quantities can be visualized to understand the economic landscape better.

## Unit Tests

Ensure the reliability and correctness of the framework by running the provided unit tests.

### Running Tests

Execute the tests using the following command:

```bash
python -m unittest discover -s ai_economist/tests
```

These tests cover various components of the central planner, including plan generation, action application, and model training.

## Contributing

Contributions are welcome! Whether you're looking to report a bug, suggest an improvement, or contribute new components, your input is valuable to us.

1. **Fork the repository:** Click the "Fork" button at the top-right of this page.
2. **Clone your fork:** 

    ```bash
    git clone https://github.com/zeynelacikgoez/ai-economist-central-planner.git
    cd ai-economist-central-planner
    ```

3. **Create a new branch:**

    ```bash
    git checkout -b feature/your-feature-name
    ```

4. **Make your changes and commit them:**

    ```bash
    git commit -m "Add your feature"
    ```

5. **Push to the branch:**

    ```bash
    git push origin feature/your-feature-name
    ```

6. **Create a Pull Request:** Navigate to your fork on GitHub and click the "Compare & pull request" button.

Please see our [contribution guidelines](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/CONTRIBUTING.md) for more details.

## License

This project is licensed under the [MIT License](LICENSE).

## Citations

If you use this code in your research, please cite us using the following BibTeX entry:

```bibtex
@misc{2004.13332,
 Author = {Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck, David C. Parkes, Richard Socher},
 Title = {The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies},
 Year = {2020},
 Eprint = {arXiv:2004.13332},
}
```

For more information and context, check out:

- [The AI Economist website](https://www.einstein.ai/the-ai-economist)
- [Blog: The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies](https://blog.einstein.ai/the-ai-economist/)
- [Blog: The AI Economist Moonshot](https://blog.einstein.ai/the-ai-economist-moonshot/)
- [Blog: The AI Economist Web Demo of the COVID-19 Case Study](https://blog.einstein.ai/ai-economist-covid-case-study-ethics/)
- [Web Demo: The AI Economist Ethical Review of AI Policy Design and COVID-19 Case Study](https://einstein.ai/the-ai-economist/ai-policy-foundation-and-covid-case-study)

## Contact

For any questions, suggestions, or collaborations, feel free to reach out:

- **Email:** ai.economist@salesforce.com
- **Slack:** Join our Slack channel [aieconomist.slack.com](https://aieconomist.slack.com) using this [invite link](https://join.slack.com/t/aieconomist/shared_invite/zt-g71ajic7-XaMygwNIup~CCzaR1T0wgA).

---

## Simulation Cards: Ethics Review and Intended Use

Please see our [Simulation Card](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/Simulation_Card_Foundation_Economic_Simulation_Framework.pdf) for a review of the intended use and ethical review of our framework.

Please see our [COVID-19 Simulation Card](https://github.com/zeynelacikgoez/ai-economist-central-planner/blob/master/COVID-19_Simulation-Card.pdf) for a review of the ethical aspects of the pandemic simulation (and as fitted for COVID-19).
