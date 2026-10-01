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


### 📊 Self-Assessment Summary

<details open>
<summary>Self-Assessment Summary</summary>

Use this checklist to gauge your readiness:

- [ ] I can write and debug Python code with confidence
- [ ] I can use Git for version control (commit, push, branches)
- [ ] I have VS Code (or similar IDE) set up with Python support
</details>

### Setup
All exercises run on a remote RunPod machine provided by the instructors, with
the repository and its dependencies already installed. Follow the
[Day 0 setup guide](day0-setup/README.md#connecting-to-your-runpod-machine) to
connect to it with VS Code.
