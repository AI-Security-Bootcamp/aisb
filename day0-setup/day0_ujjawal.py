
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

    request_url = f"https://api.github.com/users/{username}"
    response = requests.get(request_url)
    user_data = response.json()
    user_name = user_data.get("name")
    user_location = user_data.get("location")
    user_email = user_data.get("email")
    second_request_url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5"
    second_response = requests.get(second_request_url)
    print(second_response.json())
    if second_response.status_code != 200:
        return UserIntel(username=username, name=None, location=None, email=None, repo_names=[])
    else:
        repo_data = second_response.json()
        repo_names = [repo.get("name") for repo in repo_data]
        return UserIntel(username=username, name=user_name, location=user_location, email=user_email, repo_names=repo_names)
    pass
from day0_test import test_analyze_user_behavior


test_analyze_user_behavior(analyze_user_behavior)
# %%
