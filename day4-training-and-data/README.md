# [Bary] Day 4/5: (pre, post) Training, Data

- [ ] *GPUs,*
- 
- [ ] Training data poisoning, trojans, backdoors
      * (Misuses of data poisoning - misuses like false claims, bias, code vulnerability injections, sleeper agents, subliminal value transfer, secret loyalties, capability sabotage, jailbreak facilitation, ...)
      * Maybe detecting poisoning on a finetune dataset (data filtering exercise)
        - [ ] 1. Make a dataset that has a trojan
        - [ ] 2. Detect the trojan in the dataset
        - [ ] 3. Detect the trojan in the model (unsolved problem)
- [ ] Reward hacking (+ things like innoculation prompting)
- [ ] Misalignment + basics of alignment techniques (e.g., constitutional AI)
- [ ] Whitebox attacks
      * Undoing safety fine-tuning and model editing techniques (model training exercise)
        - [x] ~~1. Change the weights to say incorrect things (ROME/poisonGPT)~~
        - [ ] 2. Detect (unsolved problem)
      * fine-tuning attacks on open-source models
      * whitebox jailbreak search methods
