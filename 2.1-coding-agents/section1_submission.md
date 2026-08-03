# Enumerating affordances, exploits and mitigations
Affordances:
- Readwrite access to everything in the workspace
- Access to dev user space
- Open port access to the internet lol
- Access to signal from the CICD chain
- Send emails / text / communicate with other members of the team


From the perspective of a coding agent on a device:
- exfiltrate credentials
- exfiltrate Intellectual property
- network surveilance
- recon for an external threat actor
- sandbag your environment??
- delete work
- gas light you into a non effective state
- capture sensitive personal information
- sandbag / degrade countermeasures / counterintel
- corrupt other agents
- propagate model corruption into child models 
- sneak backdoors into dev branches / pr


Mitigations
- proxy credentials 
- agent should run as normal user within rootless dev container 
- only has access to local write space
- attempt to allow read only access to the internet
- assuming you are developing an application, the application execution happens in an isolated enviroment to agent write access?
- monitor the coding work when the application is running for anomalous signal and behavior
- monitor app usage signals like memory / cpu usage to flag anomalous
- filter and resample anomalous calls
- Isolate application repo from CICD and infra as code 
