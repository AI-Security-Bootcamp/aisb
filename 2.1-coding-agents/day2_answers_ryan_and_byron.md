# 2.1.1 enumerate affordances, exploits and mitigations

## affordances

What can the agent trigger right now, while it runs?
- Use of ssh keys and cloud credentials
- Creation of new users (backdoor accounts)
- Exfil of ssh keys and cloud credentials
- Exfil of /etc/passwd and /etc/shadow
- Exfil of local accounts on windows systems
- Exfil of password hashes
- Malicious SKILLS.md
    - Web requests
    - Downloads, C2 agent installs
    - Execute terminal commands
    - Use of API keys (access to data beyond the immediate system/lateral movement)
        - Send emails

What can it set in motion to execute after the session ends?
- Cron jobs
- p2p connections
- background jobs
- process hollowing (appearing to run as something else, or as a child of something else)
- Windows Service manipulation (unquoted service paths)
- Poisoned github repo to move to another system

Who else shares the environment: other developers, CI runners, future agent sessions?
- Human reviewers that have listed information on their scoring criteria
- Models from the same lineage that are, or can be, compromised.
- Shared gitlab/github repos
- Supply chain attacks (project dependencises usd by many people)
    - App store
    - npm package managers
    - pip packages