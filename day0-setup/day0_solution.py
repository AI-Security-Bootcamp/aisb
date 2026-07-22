# %%
"""
# Day 0 - Setup Check
Welcome to the AI Security Bootcamp! Let's start with a super simple exercise to test your setup.

<!-- toc -->

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
"""
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

# %%
"""
After you paste the code snippet above to your answer file, **run the cell to ensure it works** (typically Ctrl+Enter in VS Code).
"""
# %%
"""
## Exercise 2: Test Prerequisites
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵🔵🔵🔵

Let's verify that your development environment is properly set up with all the required tools and dependencies.

Copy-paste the code snippet below and run it to check your setup.
"""
# %%


def test_prerequisites():
    import subprocess
    import importlib
    import sys

    print("🔧 AI Security Bootcamp - Prerequisites Check")
    print("=" * 50)

    all_good = True

    # Check Python version
    def check_python_version() -> tuple[bool, str]:
        """Check if Python version is >= 3.11."""
        version = sys.version_info
        current_version = f"{version.major}.{version.minor}.{version.micro}"
        is_valid = version.major == 3 and version.minor >= 11
        return is_valid, current_version

    python_ok, python_version = check_python_version()
    status = "✅" if python_ok else "❌"
    print(f"{status} Python {python_version} {'(OK)' if python_ok else '(Need >= 3.11)'}")
    if not python_ok:
        all_good = False

    # Check Docker
    def check_docker_installed() -> bool:
        """Check if Docker is installed and accessible or if we're running in a Dev Container."""
        # Check if we're in a Dev Container (by checking if /workspaces exists)
        if os.path.isdir("/workspaces"):
            return True

        # If not in a Dev Container, check if Docker is installed
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    docker_ok = check_docker_installed()
    status = "✅" if docker_ok else "❌"
    print(f"{status} Docker {'installed' if docker_ok else 'NOT FOUND'}")
    if not docker_ok:
        all_good = False
        print("   💡 Install Docker Desktop from https://www.docker.com/products/docker-desktop/")

    # Check Git
    def check_git_configured() -> tuple[bool, str]:
        """Check if git is installed and has basic configuration."""
        try:
            # Check if git is installed
            result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False, "Git not installed"

            # # Check if user name is configured
            # result = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True, timeout=5)
            # if result.returncode != 0 or not result.stdout.strip():
            #     return False, "Git user.name not configured"

            # # Check if user email is configured
            # result = subprocess.run(['git', 'config', 'user.email'], capture_output=True, text=True, timeout=5)
            # if result.returncode != 0 or not result.stdout.strip():
            #     return False, "Git user.email not configured"

            # Check if pull.rebase is set to true
            result = subprocess.run(["git", "config", "pull.rebase"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0 or result.stdout.strip().lower() != "true":
                print("   💡 Configure with: git config pull.rebase true")
                return False, "Git missing recommended configurations"

            # Check if push.autoSetupRemote is set to true
            result = subprocess.run(
                ["git", "config", "push.autoSetupRemote"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0 or result.stdout.strip().lower() != "true":
                print("   💡 Configure with: git config --type bool push.autoSetupRemote true")
                return False, "Git missing recommended configurations"

            # Check if remote origin is set to the correct repository
            result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("   💡 Add remote with: git remote add origin git@github.com:AI-Security-Bootcamp/aisb-sg.git")
                return False, "Git remote origin not configured"

            expected_remote = "git@github.com:AI-Security-Bootcamp/aisb-sg.git"
            actual_remote = result.stdout.strip()
            if actual_remote != expected_remote:
                print(f"   💡 Current remote: {actual_remote}")
                print(f"   💡 Expected remote: {expected_remote}")
                print("   💡 Fix with: git remote set-url origin git@github.com:AI-Security-Bootcamp/aisb-sg.git")
                return False, "Git remote origin URL incorrect"

            return True, "Git properly configured"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "Git not found"

    git_ok, git_msg = check_git_configured()
    status = "✅" if git_ok else "❌"
    print(f"{status} Git: {git_msg}")
    if not git_ok:
        all_good = False
    #     if "not installed" in git_msg:
    #         print("   💡 Install Git from https://git-scm.com/downloads")
    #     else:
    #         print("   💡 Configure with: git config --global user.name 'Your Name'")
    #         print("   💡 Configure with: git config --global user.email 'your.email@example.com'")

    # Check Python packages
    def check_required_packages() -> bool:
        """Check if Python packages are installed."""
        required_packages = ["requests", "cryptography"]
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                return False

        return True

    python_packages_installed = check_required_packages()
    if python_packages_installed:
        print("✅ Python requirements are installed")
    else:
        all_good = False
        print("❌ Not all Python packages are installed")

    # Check SSH access to GitHub
    def check_github_ssh_access() -> tuple[bool, str]:
        """Check if SSH access to GitHub is working."""
        try:
            result = subprocess.run(["ssh", "-T", "git@github.com"], capture_output=True, text=True, timeout=10)
            # Print the output for debugging
            if result.stdout.strip():
                print(f"   SSH stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"   SSH stderr: {result.stderr.strip()}")

            # SSH to GitHub returns 1 on successful authentication (not 0)
            if result.returncode == 1 and "successfully authenticated" in result.stderr:
                return True, "GitHub SSH access working"
            else:
                return False, f"GitHub SSH authentication failed (return code: {result.returncode})"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, f"SSH not available or GitHub unreachable: {str(e)}"

    ssh_ok, ssh_msg = check_github_ssh_access()
    status = "✅" if ssh_ok else "❌"
    print(f"{status} GitHub SSH: {ssh_msg}")
    if not ssh_ok:
        all_good = False
        print("   💡 Configure your SSH according to the instructions in README.md")

    # Final verdict
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All prerequisites satisfied! You're ready for the bootcamp!")
    else:
        print("⚠️  Some prerequisites are missing. Please install them before proceeding.")
        print("💡 If using Dev Containers, make sure Docker is running and try reopening in container.")
        assert False, "Prerequisites check failed. Please fix the issues above."


# Run the prerequisite checks
test_prerequisites()
# %%
"""
## Exercise 3: Use requests library to make a GET request (optional)
> **Difficulty**: 🔴⚪⚪⚪⚪
> **Importance**: 🔵🔵⚪⚪⚪

In this exercise, you will use the `requests` library to make a GET request to the GitHub API and analyze a user's activity patterns.

1. Copy-paste the code snippet below into your `day0_answers.py` file. **Prepend it with `# %%` line to make it a code cell.**
2. Run the cell. The tests should fail with the default empty implementation.
3. Implement the `analyze_user_behavior` function. You can use the hints below.
4. Run the cell again to verify your implementation.
"""

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
    if "SOLUTION":
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
    else:
        # TODO: Return information about the given GitHub user
        # 1. Make a GET request to: https://api.github.com/users/{username}
        # 2. Extract name, location, and email from the response
        # 3. Make another GET request to: https://api.github.com/users/{username}/repos?sort=updated&per_page=5
        # 4. Extract repository names (limit to 5)
        # 5. Return a UserIntel object with the gathered information
        pass


@report
def test_analyze_user_behavior(solution: Callable[[str], object]):
    """Test GitHub user analysis implementation."""
    result = solution("pranavgade20")

    # Basic structure tests
    assert type(result).__name__ == "UserIntel", f"Expected UserIntel object, got {type(result)}"
    assert result.username == "pranavgade20", f"Username should be 'pranavgade20', got {result.username}"

    # Name should be populated for this public user
    assert result.name is not None, "Expected name to be found for karpathy"
    assert "Pranav" in result.name, f"Name is not correct, got {type(result.name)}"

    # Repository list tests
    assert isinstance(result.repo_names, list), f"repo_names should be list, got {type(result.repo_names)}"
    assert len(result.repo_names) <= 5, f"Should return at most 5 repos, got {len(result.repo_names)}"
    assert len(result.repo_names) > 0, "karpathy should have at least some public repositories"

    # All repo names should be non-empty strings
    for repo_name in result.repo_names:
        assert isinstance(repo_name, str), f"Repo name should be string, got {type(repo_name)}"
        assert len(repo_name) > 0, "Repo names should not be empty"

    # Test with non-existent user
    nonexistent_result = solution("this_user_definitely_does_not_exist_12345")
    assert type(result).__name__ == "UserIntel", "Should return UserIntel even for non-existent users"
    assert nonexistent_result.username == "this_user_definitely_does_not_exist_12345"
    assert nonexistent_result.name is None, "Non-existent user should have None for name"
    assert nonexistent_result.location is None, "Non-existent user should have None for location"
    assert nonexistent_result.email is None, "Non-existent user should have None for email"
    assert nonexistent_result.repo_names == [], "Non-existent user should have empty repo_names"


test_analyze_user_behavior(analyze_user_behavior)

"""
<details>
<summary>Hint 1</summary>

Use `requests.get()` to make the GET requests. The result object has a `.json()` method to parse the JSON response:

```python
user_response = requests.get(f"https://api.github.com/users/{username}")
user_data = user_response.json()
location = user_data.get("location")
```

</details>

<details>
<summary>Hint 2</summary>

Don't forget to handle error response, e.g.:

```python
if user_response.status_code != 200:
    return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])
```

</details>

<details>
<summary>Hint 3</summary>

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
</details>
"""

# %%
"""
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

"""
# %%
"""
## Further Reading
If you still want to prepare more, make sure you went through the [bootcamp prerequisites](https://docs.google.com/document/d/1PZ6-hSKEoTENxyl4vTgsM4idc8AFEfsUZgupYebfCWQ/edit?usp=sharing)!
"""
