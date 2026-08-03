# Exercise 2.1.1 

Coding agent

What can the agent trigger right now, while it is running?

- Lookup codebase, files, web search 
- Running shell scripts, bash, python scripts
- Lookup filetree
- can run subprocess 
- build and run code
- install third-party dependencies and run scripts within those 


What can it set in motion for later: things that will execute after the agent session ends?

- crontjob that kicks of later
- creat shim that you will trigger yourself
- code a trojan horse snippet within the codebase (that runs later)
- spin another agent session in the background 

Who else shares this environment? Other developers, CI runners, future agent sessions?

- depends: if it's a shared workstation, or CI environment or local environment


## Mitigations 

- user approval for specific commands 
- allow/deny list, restrict read/writes
- sandboxing 
- network egress control / nginx
- monitoring of activity and determinstic/non-deterministic blocks
