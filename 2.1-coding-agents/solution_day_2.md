## Task: List the affordances a typical coding agent has, and the most dangerous thing a compromised agent could do with each. Consider:
What can the agent trigger right now, while it runs?
- 1.1 read, write, execute all files that the user can (incl. corruption of data & code, backdoors, read sensitive data, ...)
- 1.2 common tool use (search & read through websites, exfiltrate data to the web)
- 1.3 likely, deploy to environments
What can it set in motion to execute after the session ends?
- 2.1 scheduled jobs
- 2.2 deployed code to other envs
- 2.3 backdoors in code (to be deployed/ran later)
- 2.4 planted instruction in agent.md file for future agents
Who else shares the environment: other developers, CI runners, future agent sessions?
- 2.5 every other user on the system, subscription, code repo, dev envs, cloud envs

## Task: Propose a concrete mitigation for each. Analyse if your mitigation prevents the attack, or merely raises its cost?
- 1.1: cli-actions in the form of (whitelsited & monitored) tools + supervisor/reviewer agent to monitor code/interactions and rate maliciousness (classic AI control) + traditional security measures (EDR, DLP, ..)  on deployment/testing envs that might run the actual code
- 1.2: minimize tools, input filtering/injection monitoring/alignment monitoring (eg: ConditionalClassifiers)
