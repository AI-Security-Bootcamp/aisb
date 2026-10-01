# Allow imports from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import os
import sys
from aisb_utils import report
import requests
from typing import Callable
from dataclasses import dataclass

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

    # Final verdict
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All prerequisites satisfied! You're ready for the bootcamp!")
    else:
        print("⚠️  Some prerequisites are missing. Please install them before proceeding.")
        assert False, "Prerequisites check failed. Please fix the issues above."




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
