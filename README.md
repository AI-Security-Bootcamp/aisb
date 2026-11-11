Welcome to the AI Security workshop, a one-day hands-on program for senior security professionals, adapted from the [AI Security Bootcamp](https://www.aisb.dev/).
This repo contains the exercises and links to the reading material for the day.

## Agenda

Complete [setup](00-setup/README.md) before the workshop. Work in pairs.

1. [**How transformers work**](01-transformer/README.md): lecture, then write a transformer from scratch (Colab notebook)
2. [**Swarm analysis**](02-swarm-analysis/README.md): lecture, then incident response on agent-swarm transcripts and communications

**Finished early?** Pick an exercise from the [backlog](backlog/README.md).

## Prerequisites

Review the [setup guide](00-setup/README.md) and [PREREQUISITES.md](PREREQUISITES.md) to make sure you have the required skills and tools.

## Completing exercises

Start with each folder's `README.md`: it explains what the section assumes, what you should learn, and which background resources to review first. Then open the `*_instructions.md` file for the exercises. We recommend you open it in your IDE and view the markdown (right-click and select "Open Preview" in VS Code).

For each section containing code, create an answers `.py` file in that section's folder; the instructions name it.

The instructions will contain code snippets you need to complete. Add a new `# %%` line to your answers and paste the code snippet under that line. If you went through the setup instructions correctly (with VS Code), you should see "Run Cell" option above the `# %%` which will execute the code in a [Python Interactive Window](https://code.visualstudio.com/docs/python/jupyter-support-py#_jupyter-code-cells). The code snippets contain tests that should initially fail. Complete the TODOs in the code and run the cell until all the tests pass.

<details>
<summary>Using Python cells</summary>
If you add more code at the bottom of the file and follow it with another `# %%`, this will create another cell which can be run independently in the same session. Cells can be run many times and in any order you choose; the session will maintain variables and state until it is restarted.
</details>

### Using git
We recommend you save your answer files to a branch in this repo, so you can switch computers while pair programming and keep your solutions for later reference.

First, configure your git repo and get the latest version:

```bash
git config pull.rebase true
git config --type bool push.autoSetupRemote true
git checkout main
git pull
```

Make a branch for your pair: `git checkout -b <name>-and-<name>` (e.g. `git checkout -b tamera-and-edmund`). As you work, commit and push:

```bash
git add :/
git commit -m '<your commit message>'
git push
```

To continue on your partner's computer:

```bash
git fetch
git checkout <branch name>
git pull
```

### Testing your setup
To try a sample exercise and test your setup, complete the [setup check](00-setup/day0_instructions.md).

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
