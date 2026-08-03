# %%
"""
Exercise 2.1.1: Enumerate Affordances, Exploits, and Mitigations

I initially split by "most likely" (short-term destructive) vs. "most dangerous" (subtle, persistent, spreading)

After some thought, that assumes a human-attacker threat model and doesn't really transfer to a scheming agent threat model. 

The better way to think about it is what exploits would preserve deniability for the model? Either a plausible legitimate explanation, or something that could be blamed on human error/randomness/etc. and not the model itself. 

This means mitigations should focus on observability - it's probably impossible to stop the model from completely doing bad things, but we should make it easier to detect when it does. 
"""