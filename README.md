# JUMPY-NEAT

# Project Overview
This repository showcases a mini-project developed for MML (Machine Learning) coursework, where we train an intelligent agent using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to play a game inspired by Doodle Jump. The agent learns to survive and score better through generations of evolution in a dynamic platformer environment with gravity, moving platforms, and real-time decision-making.

We're building an AI agent that learns to play a vertical scrolling platformer game (inspired by Doodle Jump) using NeuroEvolution of Augmenting Topologies (NEAT). The goal is for the agent to survive as long as possible by intelligently jumping between platforms and avoiding falling.

# Repository Structure
├── main.py                # Core game loop and NEAT integration
├── W_078_079_087_089.ipynb # Notebook with experiments and analysis
├── requirements.txt       # List of Python dependencies
├── config-5.txt           # NEAT configuration file
├── JUMPY NEAT PPT.pdf     # Final project presentation
└── README.md              # This file

### NEAT Algorithm Summary

- Evolves neural networks through evolutionary strategies.
- **Starts with minimal topologies** and progressively **complexifies** over generations.
- Maintains diversity using:
  - **Crossover** (recombination of genomes),
  - **Mutation** (adding nodes/edges or changing weights),
  - **Speciation** (grouping similar networks to protect innovation).
- Evaluates each genome based on:
  - **Game score**
  - **Survival time**

### Key Features

- **Dynamic Environment**: Platforms appear, disappear, and move.
- **Real-Time Decisions**: The agent makes decisions at every game frame.
- **Neuroevolution**: No backpropagation — evolution drives learning.
- **Speciation and Diversity**: Maintains varied strategies for robust evolution.

### Getting Started

#### Requirements
```bash
pip install -r requirements.txt
```
### Run the Game + NEAT Agent
```bash
python main.py
```

### How We're Doing It

####  Game Environment Creation
- A custom Python-based game is built using **pygame**.
- Platforms appear at random positions and the screen scrolls upward.
- The player (agent) must jump from platform to platform to stay alive.

#### NEAT Algorithm for Learning
- NEAT evolves a population of neural networks to control the agent.
- Each network (genome) receives:
  - Agent’s position,
  - Platform locations,
  - Other state features.
- Outputs a decision: **jump left**, **jump right**, or **stay**.

#### Fitness Function
Agents are rewarded based on:
- **Survival time** (longer = better),
- **Height climbed**,
- **Successful landings** on platforms.
- Poor performers are removed; strong ones are mutated and crossed over.

#### Training and Evaluation
- Over generations, NEAT evolves more efficient neural network topologies.
- The best agents improve:
  - Jump timing,
  - Movement control,
  - Survival instinct.
- All learning is through **evolution**, not backpropagation.

#### Analysis and Visualization
- A **Jupyter Notebook (.ipynb)** is provided to:
  - Analyze training curves,
  - Visualize fitness progression,
  - Interpret agent behavior.


### Screenshots

<p align="center">
  <img src="assets/jumpy_screenshot1.png" width="400"/>
  <img src="assets/jumpy_screenshot2.png" width="400"/>
</p>


### Results & Future Work

#### Achieved
- Trained agents that can jump across platforms effectively.
- Observed learning behavior across generations.

#### Future Work
- Experiment with different fitness functions and hyperparameters.
- Encourage behavioral diversity.
- Reward long survival time in scoring.

