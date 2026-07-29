### Overview

This bootcamp is intensive and fast-paced.  **The bootcamp does not teach programming or command-line basics**; you're expected to arrive with these skills already in place.

If you're unsure whether you're ready, use the self-assessment tests below. If you can answer most questions confidently and complete the practical exercises without significant struggle, you're good to go. If not, invest time in the learning resources before the bootcamp starts.

### ✅ Required Prerequisites

<details open>
<summary><b>🖥️ VS Code or Similar IDE</b></summary>

**Why this matters**: You'll use an IDE with Jupyter notebook support to run Python cells, debug code, and view markdown instructions.

**Minimum competency**:
- Open and navigate projects
- Run Python files and interactive cells (`# %%`)
- Use the integrated terminal
- Install and manage extensions
- Configure Python interpreters (select the right virtual environment)

**Self-test**:
- Can you open a folder, create a `.py` file, add a `# %%` cell, and run it?
- Do you know how to select a Python interpreter from your virtual environment?
- Can you use the built-in terminal and switch between multiple terminals?

**Resources**:
- [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- Familiarize yourself especially with the [Python Interactive Window](https://code.visualstudio.com/docs/python/jupyter-support-py) feature of VS Code
- It's very good to learn and internalize keyboard shortcuts for quickly navigating VS Code.
</details>

<details open>
<summary><b>🐍 Python</b></summary>

**Why this matters**: Coding exercises are in Python. You'll write and debug code daily, and work with virtual environments.

**Minimum competency**:
- Write functions, classes, and use common data structures (lists, dicts, sets)
- Know how to [install packages with pip](https://realpython.com/what-is-pip/) and manage virtual environments (`venv` or `conda`)
- Import and use standard libraries (`os`, `sys`, `json`, `base64`, `hashlib`)
- Read and understand error messages and stack traces
- Debug code using print statements or a debugger

**Self-test**: 
- Can you write a function that reads a JSON file, processes the data (e.g., filter, transform), and writes results to a new file?
- Can you create a virtual environment, install packages from `requirements.txt`, and activate it in your IDE?
- Can you explain what `import` does and the difference between `from module import function` vs `import module`?

**Resources**:
- **Start here**: [Python official tutorial](https://docs.python.org/3/tutorial/) (sections 3-9)
- How to create and use virtual environments, at least to the degree described by the first two sections ("How Can You Work With a Python Virtual Environment?", "How Do You Enable a Venv in Your IDE?") from this [primer from Real Python](https://realpython.com/python-virtual-environments-a-primer/)
- Practice: Solve [easy LeetCode problems](https://leetcode.com/problemset/?difficulty=EASY) and try to solve them increasingly quickly

</details>

<details open>
<summary><b>🔧 Git & Version Control</b></summary>

**Why this matters**: You'll commit your work daily, collaborate with a partner, switch between branches, and pull updates from the main repository.

**Minimum competency**:
- Clone a repository
- Create branches and switch between them
- Stage changes, commit with meaningful messages, and push to remote
- Pull changes and handle basic merge conflicts
- Understand what `.gitignore` does

**Self-test** - You can skip this if you feel comfortable with:
- Cloning a repository
- Pulling new changes
- Creating and switching between branches
- Committing changes
- Pushing branches

**Resources**:
- [An Intro to Git and GitHub for Beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)
- Alternatively, you may find this [git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf) helpful

</details>


### 🧪 Complete Setup Test

<details open>
<summary>Complete Setup Test</summary>

Before the bootcamp starts, verify your entire setup works:

**Step 1**: Create a Python virtual environment
```bash
python -m venv aisb-test
source aisb-test/bin/activate  # On Windows: aisb-test\Scripts\activate
```

**Step 2**: Install requirements
```bash
pip install -r requirements.txt
```

**Step 3**: Open VS Code, create `test.py`, and make a GET request to `https://httpbin.org/get` using the `requests` library in a Python cell (`# %%`)

**Step 4**: Run the cell and verify it runs

**Step 5**: Commit and push to a test Git repository
```bash
git init
git add test.py
git commit -m "Test setup"
git remote add origin <your-repo-url>
git push -u origin main
```

**Step 6**: Pull a Docker image and run a container
```bash
docker pull python:3.12
docker run -it python:3.12 python -c "print('Docker works!')"
```

If all steps complete without errors, you're ready! 🎉
</details>

### 📊 Self-Assessment Summary

<details open>
<summary>Self-Assessment Summary</summary>

Use this checklist to gauge your readiness:

- [ ] I can write and debug Python code with confidence
- [ ] I can use Git for version control (commit, push, branches)
- [ ] I can run Docker containers and understand basic containerization
- [ ] I have VS Code (or similar IDE) set up with Python support
</details>

### Setup
1. **Download and install [Docker desktop](https://www.docker.com/products/docker-desktop/)**
2. **Clone this repo**
    - It is recommended that you save your progress (solution files you will create throughout the bootcamp) to a branch in this repo. For that you will need to:
        - Make sure you have [an ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent) registered [with your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account), and [configured](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=mac#adding-your-ssh-key-to-the-ssh-agent) in your `.ssh/config` or ssh-agent.
        - You can test this is configured correctly with `ssh -T git@github.com`.
        - Clone the [aisb repo](https://github.com/AI-Security-Bootcamp/aisb-sg)
            ```bash
            git clone git@github.com:AI-Security-Bootcamp/aisb-sg.git
            ```
    - Alternatively, if you don't want to save your progress to a branch or just you just want to get started quickly, clone the repo with

        ```bash
        git clone https://github.com/AI-Security-Bootcamp/aisb-sg.git
        ```

### Default setup: VS Code based IDE with Dev Containers
If using VS Code base IDE, we recommend using the Dev Containers feature. This will start a Docker container with Python and all necessary dependencies already installed that your IDE will connect to. If you execute a file or open a terminal in your IDE, this will be executed inside the container while keeping the user experience of working locally (see more on [how it works](https://code.visualstudio.com/docs/devcontainers/tutorial#_how-it-works)).

- Install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- In the VS Code IDE, open Command Palette (press `F1` or select `View -> Command Palette`) and select `Dev Containers: Open Folder in Container`
- Select the directory with cloned `AI-Security-Bootcamp/aisb-sg` repo.

If you have problems with this setup on Windows, you can check out [these tips](https://code.visualstudio.com/docs/devcontainers/tips-and-tricks#_docker-desktop-for-windows-tips).

<details open>
<summary><b>If not using Dev Containers</b></summary>

If for whatever reason you decide _not_ to use Dev Containers, make sure you have the following extensions installed:

- `ms-python.python`
- `ms-python.vscode-pylance`
- `ms-toolsai.jupyter`
- `bierner.markdown-mermaid`

You will also need to set up your Python environment according to [Seting up Python environment without Dev containers](#seting-up-python-environment-without-dev-containers).
</details>

### If using a different IDE
Other IDEs are not officially supported and not recommended. If you decide to use one, you may still [be able to use dev containers](https://www.jetbrains.com/help/pycharm/connect-to-devcontainer.html). Otherwise, set up your Python environment according to [Seting up Python environment without Dev containers](#seting-up-python-environment-without-dev-containers).


### Seting up Python environment without Dev containers
You will only need to do this if you *don't* use the recommended setup with VS Code and Dev Containers.

<details open>
<summary>Expand instructions</summary>

For most exercises, you need a Python environment with Python >= 3.11 and the dependencies from `requirements.txt` installed. If an exercise needs a more complicated setup, it will be described in its instructions.

You can set up the Python environment with these steps:

1. [Install miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
2. Verify conda was installed and activated by running `conda --version`
3. Create and activate a new environment:
    
    ```bash
    conda create --name aisb python=3.11
    conda activate asib
    ```
4. Navigate to this directory and install requirements:

    ```bash
    pip install -r requirements.txt
    ```
5. Make sure that the new conda environment is activated in your IDE. You can get the correct path to Python executable with

    ```bash
    conda run -n aisb which python
    ```

</details>