Welcome to the [AI Security Bootcamp](https://www.aisb.dev/) (AISB)! AISB is a 7-day intensive program for security professionals shaping how we secure emerging AI systems. This repo contains the exercises and links to the reading material you will go through during the bootcamp.

## What you should know and learn

Day 0 should establish that each participant can run the course, save their work,
and reach every external service before instructional time begins.

**Suggested time:** 45–60 minutes

**Exercises:** [Open the participant setup instructions](day0_instructions.md)

| Area | Prerequisites coming in | Main learnings going out |
| --- | --- | --- |
| Engineering | - | Pass the Docker, Python-cell, HTTP-request, and Git-workflow checks and identify which component failed when a check does not pass |
| ML | - | Classify each later lab as using local CPU, remote GPU, or a hosted API |
| Security | SSH key use and secret handling | Store API and SSH credentials without committing them and distinguish connectivity, authentication, and authorization failures |
| Theory | - | Locate the instructions, answer file, tests, generated artifacts, and support route for an exercise |

### Background

- Complete the checks in the [participant setup instructions](day0_instructions.md).
- Install [Docker](https://docs.docker.com/engine/install/), [Git](https://git-scm.com/downloads), [Python](https://www.python.org/downloads/), and [VS Code with Jupyter support](https://code.visualstudio.com/docs/datascience/jupyter-notebooks), and confirm that you can access GitHub.
- Obtain the event-provided API and GPU credentials before the live programme.

## Prerequisites

Review the prerequisites and setup instructions below to ensure you have the required skills and tools to complete the bootcamp.

## In-person instructions
If you're attending the bootcamp in person, you will spend most of the days pair programming with your assigned partner on the exercises for the given day.

Before the first day, make sure you have completed the setup instructions below.


### Completing exercises
To view the instructions for a section, navigate to the respective directory (e.g., `1.1-llm-internals`) and open the `*_instructions.md` file located there. We recommend you open it in your IDE and view the markdown (right-click and select "Open Preview" in VS Code).

The recommended way to complete the exercises is to make a new `.py` file (suggested name: `day#_answers.py`) in the directory for the day.

The instructions will contain code snippets you need to complete. Add a new `# %%` line to your answers and paste the code snippet under that line. If you went through the setup instructions correctly (with VS Code), you should see "Run Cell" option above the `# %%` which will execute the code in a [Python Interactive Window](https://code.visualstudio.com/docs/python/jupyter-support-py#_jupyter-code-cells). The code snippets contain tests that should initially fail. Complete the TODOs in the code and run the cell until all the tests pass.

<details>
<summary>Using Python cells</summary>
If you add more code at the bottom of the file and follow it with another `# %%`, this will create another cell which can be run independently in the same session. Cells can be run many times and in any order you choose; the session will maintain variables and state until it is restarted.
</details>

### Using git
We recommend you save your progress for each day (the answer files you will create with your assigned partner) to a branch in this repo. This will make it possible for you to switch between computers you will use while pair programming, and make your solution available for you to reference later.

First, configure your git repo with:

```bash
git config pull.rebase true
git config --type bool push.autoSetupRemote true
```

**Every day in the morning**, make sure you have the latest version of the repo:
```bash
git checkout main
git pull
```

Make a branch for the day: `git checkout -b <branch name>`, where your branch name should follow the convention `day#/<name>-and-<name>`. For example, if Tamera and Edmund were pairing on the day 3 content, the command would be `git checkout -b day3/tamera-and-edmund`. If you share a first name with someone else in the program, use a unique nickname of your choice for disambiguation.

Create a new file for your answers (see [completing exercises](#completing-exercises) above) and work through the material with your partner.

As you work, commit changes to your branch and push them to the repo. To make and push a commit:

```bash
git add :/
git commit -m '<your commit message>'
git push
```

If you want to switch what computer you work on with your partner, they can check out the latest version of the branch with:

```bash
git fetch
git checkout <branch name>
git pull
```


### Testing your setup
If you'd like to try a sample exercise and test your setup, go ahead and complete [day0](day0_instructions.md)!
