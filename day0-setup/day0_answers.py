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

from day0_test import test_prerequisites


# Run the prerequisite checks
test_prerequisites()

# %%

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
    # Look up the user's public profile.
    user_response = requests.get(f"https://api.github.com/users/{username}")
    if user_response.status_code != 200:
        # Unknown (or unreachable) user: return an empty profile rather than raising.
        return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])

    user_data = user_response.json()

    # Fetch the 5 most recently updated repos to gauge what the user is working on now.
    repos_response = requests.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5")
    repo_names = []
    if repos_response.status_code == 200:
        repo_names = [repo["name"] for repo in repos_response.json()[:5]]

    return UserIntel(
        username=username,
        name=user_data.get("name"),
        location=user_data.get("location"),
        email=user_data.get("email"),
        repo_names=repo_names,
    )


print(analyze_user_behavior())

from day0_test import test_analyze_user_behavior


test_analyze_user_behavior(analyze_user_behavior)
