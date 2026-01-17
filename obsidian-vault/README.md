# Obsidian Vault for Cuttle DQN Thesis

This is an Obsidian vault containing organized notes for your thesis on the Cuttle card game Deep Q-Network (DQN) reinforcement learning project.

## Getting Started

### Opening in Obsidian

1. Open Obsidian
2. Click "Open folder as vault"
3. Navigate to this directory: `obsidian-vault/`
4. Select the folder

The vault will open with all notes organized in folders.

## Structure

```
obsidian-vault/
├── 00-Index/
│   └── Home.md                    # Start here! Main index page
├── 01-Project/                    # Project-specific documentation
│   ├── Project Overview.md
│   ├── CuttleEnvironment.md
│   └── Observation Space.md
├── 02-Machine-Learning/           # ML fundamentals
│   └── Machine Learning Overview.md
├── 03-Reinforcement-Learning/     # RL concepts
│   ├── Reinforcement Learning.md
│   ├── Q-Learning.md
│   ├── Deep Q-Network.md
│   ├── Self-Play.md
│   ├── Experience Replay.md
│   ├── Epsilon-Greedy Exploration.md
│   └── Reward Engineering.md
├── 04-Neural-Networks/            # Neural network concepts
│   ├── Neural Network Basics.md
│   ├── Activation Functions.md
│   └── Optimization Algorithms.md
├── 05-Technologies/               # Frameworks and tools
│   ├── PyTorch.md
│   └── Gymnasium.md
└── 06-Training/                   # Training techniques
    ├── Hyperparameters.md
    └── Gradient Clipping.md
```

## Navigation

- **Start with [[Home]]**: Contains table of contents and quick links
- **Use WikiLinks**: Double brackets `[[Note Name]]` create links between notes
- **Tags**: Notes are tagged by topic (e.g., `#machine-learning`, `#reinforcement-learning`)
- **Search**: Use Obsidian's search to find concepts quickly

## Features

### WikiLinks

All notes use `[[WikiLinks]]` to cross-reference related concepts. Click on any link to navigate.

### Tags

Notes are tagged for easy filtering:
- `#project` - Project-specific
- `#machine-learning` - ML concepts
- `#reinforcement-learning` - RL concepts
- `#neural-networks` - NN concepts
- `#technology` - Frameworks/tools
- `#training` - Training techniques

### Frontmatter

Each note has YAML frontmatter with:
- `title`: Note title
- `tags`: Topic tags
- `created`: Creation date
- `related`: Related note names

## Adding Your Own Notes

1. Create new markdown file (`.md`) in appropriate folder
2. Add frontmatter at top:
   ```yaml
   ---
   title: Your Note Title
   tags: [tag1, tag2]
   created: YYYY-MM-DD
   related: [Related Note 1, Related Note 2]
   ---
   ```
3. Use `[[WikiLinks]]` to link to existing notes
4. Add tags with `#tag-name`

## Tips for Thesis Writing

- **Use the graph view**: Visualize connections between concepts
- **Search frequently**: Find concepts quickly with `Ctrl+P` (Cmd+P on Mac)
- **Create your own notes**: Add thesis-specific notes as you write
- **Link everything**: Use WikiLinks to build knowledge graph

## Note Organization

Notes are organized by topic:
- **00-Index**: Navigation and overview
- **01-Project**: Your specific project details
- **02-Machine-Learning**: General ML concepts
- **03-Reinforcement-Learning**: RL-specific concepts
- **04-Neural-Networks**: Neural network fundamentals
- **05-Technologies**: Tools and frameworks
- **06-Training**: Training methodologies

## Getting Help

- **Obsidian Help**: https://help.obsidian.md/
- **WikiLink Syntax**: `[[Note Name|Display Text]]` for custom display text
- **Tag Syntax**: `#tag-name` creates tag, `#tag-name/subtag` creates nested tag

---

*Happy note-taking and thesis writing!*
