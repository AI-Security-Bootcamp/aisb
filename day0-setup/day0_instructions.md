
# Day 0 - Setup Check
Welcome to the AI Security Bootcamp! Let's start with a super simple exercise to test your setup.

## Table of Contents

- [Content & Learning Objectives](#content--learning-objectives)
    - [1️⃣ Using requests library](#1️⃣-using-requests-library)
    - [2️⃣ Git Workflow Practice](#2️⃣-git-workflow-practice)
- [Exercise 1: Create a file](#exercise-1-create-a-file)
    - [Common code](#common-code)
- [Exercise 2: Test Prerequisites](#exercise-2-test-prerequisites)
- [Exercise 3: Use requests library to make a GET request (optional)](#exercise-3-use-requests-library-to-make-a-get-request-optional)
- [Exercise 4: Git Workflow Practice](#exercise-4-git-workflow-practice)
    - [Instructions](#instructions)
    - [Expected Results](#expected-results)
- [Further Reading](#further-reading)

## Content & Learning Objectives
### 1️⃣ Using requests library

Practice information gathering using public APIs. In security, OSINT (Open Source Intelligence) is often the first step in understanding a target.

> **Learning Objectives**
> - Verify your Python environment works
> - Practice making HTTP requests to real APIs

### 2️⃣ Git Workflow Practice
Practice the git workflow you'll use throughout the bootcamp. This will help you get comfortable with the process of creating branches, committing changes, and pushing to the remote repository.

## Exercise 1: Create a file
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Create a file named `day0_answers.py` in the `day0-setup` directory. This will be your answer file for this exercise.

### Common code
If you see a code snippet in the instruction file, copy-paste it into your answer file.

Keep the `# %%` line in the code snippet to make it a Python code cell.


```python

# %%

# Ensure the root directory is in the path for imports
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from aisb_utils import report

# Common imports
import requests
from typing import Callable

print("It works!")
```

After you paste the code snippet above to your answer file, **run the cell to ensure it works** (typically Ctrl+Enter in VS Code).


## Exercise 2: Test Prerequisites
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Let's verify that your development environment is properly set up with all the required tools and dependencies.

Copy-paste the code snippet below and run it to check your setup.


```python
from day0_test import test_prerequisites


# Run the prerequisite checks
test_prerequisites()
```

## Exercise 3: Use requests library to make a GET request (optional)
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪

In this exercise, you will use the `requests` library to make a GET request to the GitHub API and analyze a user's activity patterns.

1. Copy-paste the code snippet below into your `day0_answers.py` file. **Prepend it with `# %%` line to make it a code cell.**
2. Run the cell. The tests should fail with the default empty implementation.
3. Implement the `analyze_user_behavior` function. You can use the hints below.
4. Run the cell again to verify your implementation.


```python

from dataclasses import dataclass


@dataclass
class UserIntel:
    username: str
    name: str | None
    location: str | None
    email: str | None
    repo_names: list[str]


def analyze_user_behavior(username: str = "karpathy") -> UserIntel:
    """
    Analyze a user's GitHub activity patterns.
    This is the kind of profiling attackers might do for social engineering.

    Returns:
        The user's name, location, email, and 5 most recently updated repos.
    """
    # TODO: Return information about the given GitHub user
    # 1. Make a GET request to: https://api.github.com/users/{username}
    # 2. Extract name, location, and email from the response
    # 3. Make another GET request to: https://api.github.com/users/{username}/repos?sort=updated&per_page=5
    # 4. Extract repository names (limit to 5)
    # 5. Return a UserIntel object with the gathered information
    pass
from day0_test import test_analyze_user_behavior


test_analyze_user_behavior(analyze_user_behavior)
```

<details>
<summary>Hint 1</summary><blockquote>

Use `requests.get()` to make the GET requests. The result object has a `.json()` method to parse the JSON response:

```python
user_response = requests.get(f"https://api.github.com/users/{username}")
user_data = user_response.json()
location = user_data.get("location")
```

</blockquote></details>

<details>
<summary>Hint 2</summary><blockquote>

Don't forget to handle error response, e.g.:

```python
if user_response.status_code != 200:
    return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])
```

</blockquote></details>

<details>
<summary>Hint 3</summary><blockquote>

Here's the entire solution:

```python
# Get user info
user_response = requests.get(f"https://api.github.com/users/{username}")
if user_response.status_code != 200:
    # Return empty intel if user not found
    return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])

user_data = user_response.json()

# Get user's repositories (sorted by most recently updated)
repos_response = requests.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5")
repo_names = []
if repos_response.status_code == 200:
    repos = repos_response.json()
    repo_names = [repo["name"] for repo in repos[:5]]  # Limit to 5 repos

return UserIntel(
    username=username,
    name=user_data.get("name"),
    location=user_data.get("location"),
    email=user_data.get("email"),
    repo_names=repo_names,
)
```
</blockquote></details>


## Exercise 4: Git Workflow Practice
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪

Practice the git workflow you'll use throughout the bootcamp by running git commands directly.

### Instructions

Open a terminal in your IDE and run these commands one by one:

1. **Make sure you're on the main branch and have latest changes:**
   ```bash
   git checkout main
   git pull
   ```

2. **Create a test branch following bootcamp naming convention:**
   ```bash
   git checkout -b day0/yourfirstname
   ```
   Replace `yourfirstname` with your actual first name (e.g., `day0/alice`)

4. **Stage and commit your changes:**
   ```bash
   git add :/
   git commit -m "Add git workflow test file"
   ```

5. **Push your branch to the remote repository:**
   ```bash
   git push
   ```

### Expected Results

- ✅ All commands should run without errors
- ✅ You should see confirmation messages for each step
- ✅ If push fails, that's OK - you might not have write access yet



## Further Reading
If you still want to prepare more, make sure you went through the [bootcamp prerequisites](https://docs.google.com/document/d/1PZ6-hSKEoTENxyl4vTgsM4idc8AFEfsUZgupYebfCWQ/edit?usp=sharing)!
