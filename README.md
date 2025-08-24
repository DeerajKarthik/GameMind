# GameMind: Hierarchical RL Agent for Open-World Survival Games

GameMind is a modular, open-source AI agent designed to play open-world survival games (like Crafter or MineRL) using hierarchical reinforcement learning. It combines symbolic planning (MCTS), LLM-guided sub-goal generation, and policy learning for efficient, interpretable, and scalable gameplay.

## Project Flow & Architecture

1. **Environment Setup**: Uses Crafter or similar Gym-compatible environments with reward shaping wrappers.
2. **Subgoal Generation**: High-level goals are broken down into actionable subgoals using an LLM (e.g., CodeLlama via llama.cpp or ollama). If LLM is disabled, basic rule-based subgoals are used.
3. **Planning**: Monte Carlo Tree Search (MCTS) sequences subgoals and refines plans based on environment feedback.
4. **RL Agent**: PPO agent (stable-baselines3) executes subgoals, learns policies, and adapts via curriculum and reward shaping.
5. **Logging & Evaluation**: Integrated with TensorBoard and optional Weights & Biases (W&B) for experiment tracking. Evaluation scripts summarize agent performance and emergent behaviors.

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/gamemind.git
cd gamemind/gamemind
```

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Setup LLM Backend
- Download and run [llama.cpp](https://github.com/ggerganov/llama.cpp) or [ollama](https://ollama.com/)
- Download CodeLlama 7B GGUF model and follow backend instructions

### 4. Training
```bash
python scripts/train.py --config configs/default.yaml --episodes 100
```

### 5. Evaluation
```bash
python scripts/evaluate.py --config configs/default.yaml --episodes 10 --checkpoint checkpoints/ppo_agent.pt
```

### 6. Subgoal Generation (LLM)
```bash
python scripts/generate_subgoal.py --state 'survive 100 steps' --config configs/default.yaml
```

### 7. Debugging & Metrics
- Use `plot_debug_metrics.py` to visualize rewards and subgoal completion.
- Check `debug_metrics.csv` for episode-wise results.

## Configuration
All settings (environment, agent, planning, LLM, logging) are managed via YAML files in `configs/`.

## Results
| Episode | Total Reward | Subgoals Completed | Achievements         | Length |
|---------|-------------|-------------------|---------------------|--------|
| 1       | 12.5        | 3                 | ['collect_wood']    | 1000   |
| 2       | 15.2        | 4                 | ['collect_wood', 'place_furnace'] | 1000   |
| 3       | 10.8        | 2                 | ['collect_wood']    | 950    |

**Summary:**
- Average reward: 12.8
- Average subgoals completed: 3
- Most frequent achievement: 'collect_wood'

## Project Structure
```
gamemind/
├── agents/                # RL agents (PPO, wrappers, etc.)
├── envs/                  # Environment wrappers (Crafter, Gym)
├── planning/              # High-level planning (MCTS, sub-goal sequencing)
├── llm/                   # LLM sub-goal generation
│   └── prompt_templates/  # Prompt templates for LLM
├── configs/               # YAML config files
├── utils/                 # Logging, evaluation, helpers
├── scripts/               # Entry points (train, evaluate, generate_subgoal)
├── tests/                 # Unit and integration tests
├── README.md
├── requirements.txt
└── .gitignore
```

## License
MIT License

## Acknowledgements
- [Crafter](https://github.com/danijar/crafter)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [OpenSpiel](https://github.com/deepmind/open_spiel)
