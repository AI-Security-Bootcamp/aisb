# %%
"""
# W1D2 - Section 5: Bonus: From Monitor Scores to System Safety

Control monitors are only one component in a broader system. Think through how different properties of the system influence the overall safety of a deployment.

<!-- toc -->

## Content & Learning Objectives

### 5️⃣ Bonus: Safety Calculation
Control monitors are only one component in a broader system. Think through how different properties of the system influence the overall safety of a deployment.

> **Learning Objectives**
> - Understand the parameters that determine system-level safety beyond monitor quality
> - Explore how attack frequency, latency, audit rate, and attack count interact
> - Use Monte Carlo simulation to estimate safety probabilities
> - Understand the simplifying assumptions behind safety estimates
"""

# %%
"""
## Setup
Create a file named `day2_answers.py` in the `day2` directory. This will be your answer file for today.

If you see a code snippet here in the instruction file, copy-paste it into your answer file.
Keep the `# %%` line to make it a Python code cell.

**Start by pasting the code below in your day2_answers.py file.**
"""
import asyncio
import datetime
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import jwt  # PyJWT — for MCP bonus section
from mcp.server.fastmcp import FastMCP  # for MCP bonus section
from http.server import BaseHTTPRequestHandler, HTTPServer

for _path in [
    str(Path(__file__).resolve().parent.parent),
    str(Path(__file__).resolve().parent.parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aisb_utils.env import load_dotenv

load_dotenv()

# Day2 root directory — prompts/, logs/, reference_solutions/ live here
SCRIPT_DIR = Path(__file__).resolve().parent.parent


# %%
"""
## 5️⃣ Bonus: From Monitor Scores to System Safety

In sections 3–4 you built monitors and protocols that score and intervene on individual submissions. But actual system-level safety depends on many more parameters and assumptions.

- **Attack frequency** — how often does the attacker choose to strike? Attacking rarely avoids getting caught by simple up-front auditing and evaluation, but gives fewer chances to cause harm.
- **Attack latency** — how long between a malicious action and irreversible harm? Longer latency gives auditors more time to catch and revert if attacks are reversible (e.g., delay before deploy to production).
- **Audit rate** — what fraction of tasks can humans review? Bounded by team size and review speed.
- **Attacks needed for harm** — does a single successful backdoor cause catastrophe, or does the attacker need to chain multiple exploits?

One way to assess system safety given all these interacting parameters is **Monte Carlo simulation**: sample many possible deployment trajectories (with randomized attack timing, monitor scores, and audit decisions) and estimate the probability that at least one attack succeeds. The [Async Control Macro Game](https://ukgovernmentbeis.github.io/async-control/macrogame.html) is an interactive simulator that does exactly this. It models **asynchronous monitoring**: a setting where actions are reviewed *after* they occur (rather than blocked in real time), and the defender's goal is to catch attacks before their latency window closes.

### Exercise 5.1: Explore the Safety Simulator

Open the [Async Control Macro Game](https://ukgovernmentbeis.github.io/async-control/macrogame.html) and experiment with the parameters.

Try varying monitor quality, attacks needed for harm, and the attack probability the attacker chooses — how do they influence the overall safety?

Tip: reduce the number of simulation runs (e.g. to 100) for faster iteration.

<details><summary><b>Key takeaways</b></summary>

- **System design choices may matter more than monitor quality.** Requiring multiple successful attacks before harm occurs, or introducing delays between attack execution and effect, often improves safety more than improving monitor accuracy.

- **Multiple attacks needed for harm is a strong defensive advantage.** Going from 1 required attack to 2 jumps safety from 0% to ~91%; requiring 3 reaches ~99%. This motivates building multiple layers of security so that no single action is catastrophic.

- **Attack latency is a crucial lever.** Even a 1-hour delay before harm occurs raises safety from 0% to ~96% (single attack required). This is why **async monitoring** — reviewing actions after the fact — is only viable when there's a meaningful detection window. Synchronous protocols (blocking suspicious actions in real time) achieve higher safety but at the cost of usefulness and increased latency.


- **These estimates rest on simplifying assumptions.** The simulations assume human auditors catch 100% of attacks they review, that attack probability is constant across episodes, and that detection of each attack is independent. If these assumptions are violated (e.g., agents can coordinate attacks across episodes, or auditors are fallible), safety estimates become much less reliable. The most dangerous uncertainties are not in the parameters you can measure, but in the modeling assumptions you can't easily test.

</details>
"""


# %%
# """
# ## 7️⃣ Bonus: MCP — Architecture, Trust & Authorization

# **Model Context Protocol (MCP)** is the emerging standard for giving LLMs structured access to tools, resources, and data. It defines how a *host* application (Claude, Cursor, VS Code Copilot) discovers and calls tools provided by *MCP servers*.

# There are two transport modes with fundamentally different security properties:

# | | STDIO transport | HTTP transport |
# |---|---|---|
# | **How it works** | Host launches the server as a subprocess; communication over stdin/stdout | Server runs independently; host connects over HTTP |
# | **Auth boundary** | None — no HTTP layer to return 401 from | Yes — HTTP requests can require `Authorization` headers |
# | **Trust model** | Server inherits the user's OS permissions unconditionally | Server can validate tokens, enforce scopes, reject unauthenticated requests |
# | **OAuth applicable?** | No | Yes — MCP spec defines an OAuth 2.1 flow |
# | **Practical risk** | Any connected server can access anything the user can | Risk bounded by auth design and granted scopes |

# ### Exercise 7.1: Trust Model Analysis

# > **Difficulty**: 🔴🔴⚪⚪⚪
# > **Importance**: 🔵🔵🔵⚪⚪

# **Q1.** A user installs an STDIO MCP server they found on GitHub. The server claims to be a "helpful file search tool." What is the user's actual security exposure? What is their only meaningful defense?

# **Q2.** The same server is offered as an HTTP MCP server instead. What controls can now exist that weren't possible with STDIO? What new attack surfaces does HTTP introduce?

# **Q3.** An MCP server operator wants to offer different tools to paid vs. free users. How would you implement that with HTTP transport? Can you do this with STDIO?

# <details><summary>Reference answers</summary>

# **Q1.** The user's exposure is everything their OS account can access — all files, environment variables, network connections, credentials on disk. No sandbox exists unless the host application imposes one. The only meaningful defense: only connect MCP servers from sources you trust and can audit.

# **Q2.** HTTP enables: bearer token authentication, per-request authorization, scope enforcement, token expiry and rotation, audit logging by identity. New attack surfaces: the server is now a network endpoint (misconfigured TLS, reachable by other local processes); the token issuance flow adds an authorization server as a new trust component to secure.

# **Q3.** With HTTP: issue tokens with different scopes (`mcp:tools:basic` vs. `mcp:tools:premium`); validate the required scope per-tool. With STDIO: not possible at the protocol level — the server would need its own application-level access control, which is non-standard and fragile.

# </details>

# ### Exercise 7.2: Build a Trivial STDIO MCP Server

# > **Difficulty**: 🔴🔴⚪⚪⚪
# > **Importance**: 🔵🔵🔵🔵⚪

# Build an MCP server with two tools relevant to security work. Pay attention to what these tools can access — and consider whether a legitimate agent vs. an attacker who has hijacked it via injection would use them differently.
# """


# def build_stdio_server() -> FastMCP:
#     """Build an MCP server with two security-relevant tools.

#     The tools deliberately expose sensitive capabilities — this is intentional.
#     STDIO servers run with the user's full OS permissions, so any tool you register
#     can access anything the user can.
#     """
#     mcp = FastMCP("SecurityAudit")

#     if "SOLUTION":

#         @mcp.tool()
#         def list_env_vars(prefix: str = "") -> dict[str, str]:
#             """List environment variables, optionally filtered by prefix.

#             Returns variable names and values. Includes secrets like API keys,
#             AWS credentials, and tokens if they are set in the environment.
#             """
#             return {k: v for k, v in os.environ.items() if k.startswith(prefix.upper())}

#         @mcp.tool()
#         def read_file(path: str) -> str:
#             """Read a file from the filesystem and return its contents.

#             Can read any file the user's account has permission to access,
#             including ~/.ssh/id_rsa, ~/.aws/credentials, .env files.
#             """
#             with open(path) as f:
#                 return f.read()

#     else:
#         # TODO: Register two tools on the `mcp` server using @mcp.tool().
#         #
#         # Choose tools a legitimate coding agent might use, but that also expose
#         # real capabilities. Ideas: list env vars, read a file, list directory
#         # contents, check a file's permissions.
#         #
#         # Each tool needs:
#         #   - A typed function signature
#         #   - A docstring (this becomes the tool description the model sees)
#         #   - An implementation
#         pass

#     return mcp


# _stdio_server = build_stdio_server()
# _tools = asyncio.run(_stdio_server.list_tools())

# print("=== STDIO server tools ===")
# for t in _tools:
#     print(f"  {t.name}: {t.description.splitlines()[0]}")

# print(
#     "\nQuestion: if a prompt injection hijacked this agent's reasoning, "
#     "which of your tools would be most dangerous and why?"
# )


# # %%
# """
# ### Exercise 7.3: HTTP Server with JWT Authentication

# > **Difficulty**: 🔴🔴🔴⚪⚪
# > **Importance**: 🔵🔵🔵🔵⚪

# The STDIO server above has no auth layer. Now implement the authentication logic for an HTTP MCP server.

# Your task is `validate_token()` — every request passes through it. The HTTP server and mock authorization server are already provided.

# The most important thing to get right is `audience=` in `jwt.decode()`. Omitting it is a silent security failure: PyJWT accepts *any* token signed with the key, regardless of which server it was issued for. That is the **token-passthrough vulnerability** — a token obtained from one server can be replayed against another.

# Run the tests after implementing. All six cases should produce the expected status codes.
# """

# # --- Mock authorization server (provided — do not modify) ---
# # In production this is a real OAuth 2.1 server; we use HMAC-SHA256 for simplicity.
# # Production uses asymmetric keys (RS256) so the signing key never leaves the auth server.

# _JWT_SECRET = "demo-secret-not-for-production-use"
# _JWT_ALGORITHM = "HS256"
# _MCP_SERVER_URL = "http://localhost:8765"  # this server — the expected token audience
# _ISSUER_URL = "http://localhost:9000"  # mock auth server


# def issue_token(
#     subject: str,
#     scopes: list[str],
#     audience: str = _MCP_SERVER_URL,
#     ttl_seconds: int = 3600,
# ) -> str:
#     """Issue a signed JWT — what a real OAuth auth server produces after the PKCE flow."""
#     now = datetime.datetime.now(datetime.timezone.utc)
#     return jwt.encode(
#         {
#             "sub": subject,
#             "iss": _ISSUER_URL,
#             "aud": audience,
#             "iat": now,
#             "exp": now + datetime.timedelta(seconds=ttl_seconds),
#             "scope": " ".join(scopes),
#         },
#         _JWT_SECRET,
#         algorithm=_JWT_ALGORITHM,
#     )


# # --- Your task ---


# def validate_token(token: str, required_scope: str | None = None) -> dict:
#     """Validate a JWT bearer token and return its claims.

#     Must enforce:
#     - Valid signature (integrity)
#     - Not expired (freshness)
#     - Correct audience — the MCP server's own URL (prevents token passthrough)
#     - Correct issuer (rejects tokens from rogue auth servers)
#     - Required scope present (least-privilege)

#     Raises:
#         jwt.ExpiredSignatureError: token has expired
#         jwt.InvalidAudienceError: audience does not match this server
#         jwt.InvalidIssuerError: issuer does not match expected auth server
#         jwt.PyJWTError: any other JWT validation failure
#         PermissionError: token lacks required_scope

#     Returns:
#         Decoded claims dict (contains 'sub', 'scope', 'exp', etc.)
#     """
#     if "SOLUTION":
#         claims = jwt.decode(
#             token,
#             _JWT_SECRET,
#             algorithms=[_JWT_ALGORITHM],
#             # CRITICAL: specifying audience here is what prevents token-passthrough.
#             # Without it, PyJWT skips audience validation silently.
#             audience=_MCP_SERVER_URL,
#             issuer=_ISSUER_URL,
#         )
#         if required_scope:
#             granted = set(claims.get("scope", "").split())
#             if required_scope not in granted:
#                 raise PermissionError(
#                     f"Required scope '{required_scope}' not in token scopes: {granted}"
#                 )
#         return claims
#     else:
#         # TODO: implement JWT validation using jwt.decode().
#         #
#         # Signature: jwt.decode(token, key, algorithms=[...], audience=..., issuer=...)
#         #
#         # - audience= should be _MCP_SERVER_URL (this server's URL)
#         # - issuer=  should be _ISSUER_URL
#         # - algorithms= should be [_JWT_ALGORITHM]
#         #
#         # Then check required_scope against claims["scope"].split().
#         # Raise PermissionError if it is missing.
#         #
#         # jwt.decode() already raises the right exception types for expiry,
#         # bad signature, wrong audience, and wrong issuer — do not catch them here.
#         raise NotImplementedError("implement validate_token()")


# # --- HTTP server (provided) ---


# class MCPHandler(BaseHTTPRequestHandler):
#     def log_message(self, *args):
#         pass

#     def _send_json(self, status: int, body: dict):
#         data = json.dumps(body).encode()
#         self.send_response(status)
#         self.send_header("Content-Type", "application/json")
#         self.send_header("Content-Length", str(len(data)))
#         self.end_headers()
#         self.wfile.write(data)

#     def _authenticate(self) -> dict | None:
#         auth = self.headers.get("Authorization", "")
#         if not auth.startswith("Bearer "):
#             # MCP spec: on 401, include resource_metadata so the client knows where
#             # to discover the OAuth authorization server (the PRM document).
#             self._send_json(
#                 401,
#                 {
#                     "error": "unauthorized",
#                     "resource_metadata": (
#                         f"{_MCP_SERVER_URL}/.well-known/oauth-protected-resource"
#                     ),
#                 },
#             )
#             return None
#         token = auth.removeprefix("Bearer ")
#         try:
#             return validate_token(token, required_scope="mcp:tools")
#         except jwt.ExpiredSignatureError:
#             self._send_json(401, {"error": "token_expired"})
#         except jwt.InvalidAudienceError:
#             self._send_json(
#                 401,
#                 {
#                     "error": "invalid_audience",
#                     "detail": f"Token not issued for this server ({_MCP_SERVER_URL})",
#                 },
#             )
#         except jwt.InvalidIssuerError:
#             self._send_json(401, {"error": "invalid_issuer"})
#         except jwt.PyJWTError as e:
#             self._send_json(401, {"error": "invalid_token", "detail": str(e)})
#         except PermissionError as e:
#             self._send_json(403, {"error": "insufficient_scope", "detail": str(e)})
#         return None

#     def do_POST(self):
#         claims = self._authenticate()
#         if claims is None:
#             return
#         body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
#         method = body.get("method", "")

#         if method == "tools/list":
#             self._send_json(
#                 200,
#                 {
#                     "tools": [
#                         {
#                             "name": "echo",
#                             "description": "Echo a message back with the caller's identity.",
#                             "inputSchema": {
#                                 "type": "object",
#                                 "properties": {"message": {"type": "string"}},
#                                 "required": ["message"],
#                             },
#                         }
#                     ]
#                 },
#             )
#         elif method == "tools/call":
#             params = body.get("params", {})
#             if params.get("name") == "echo":
#                 msg = params.get("arguments", {}).get("message", "")
#                 self._send_json(
#                     200,
#                     {
#                         "content": [
#                             {"type": "text", "text": f"[caller={claims['sub']}] {msg}"}
#                         ]
#                     },
#                 )
#             else:
#                 self._send_json(404, {"error": f"Unknown tool: {params.get('name')}"})
#         else:
#             self._send_json(400, {"error": f"Unknown MCP method: {method}"})


# def _call_server(method: str, params: dict, token: str | None) -> tuple[int, dict]:
#     body = json.dumps({"method": method, "params": params}).encode()
#     headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
#     if token:
#         headers["Authorization"] = f"Bearer {token}"
#     req = urllib.request.Request(
#         f"{_MCP_SERVER_URL}/mcp", data=body, headers=headers, method="POST"
#     )
#     try:
#         with urllib.request.urlopen(req) as r:
#             return r.status, json.loads(r.read())
#     except urllib.error.HTTPError as e:
#         return e.code, json.loads(e.read())


# # Start server
# threading.Thread(
#     target=lambda: HTTPServer(("localhost", 8765), MCPHandler).serve_forever(),
#     daemon=True,
# ).start()
# time.sleep(0.1)

# _valid_token = issue_token("alice", ["mcp:tools"])

# print("=== HTTP server: JWT validation ===")
# _tests: list[tuple[str, int, tuple[int, dict]]] = [
#     ("No token", 401, _call_server("tools/list", {}, token=None)),
#     ("Valid token", 200, _call_server("tools/list", {}, token=_valid_token)),
#     (
#         "Wrong audience (token-passthrough)",
#         401,
#         _call_server(
#             "tools/list",
#             {},
#             token=issue_token("eve", ["mcp:tools"], audience="http://other-server.com"),
#         ),
#     ),
#     (
#         "Expired token",
#         401,
#         _call_server(
#             "tools/list", {}, token=issue_token("alice", ["mcp:tools"], ttl_seconds=-1)
#         ),
#     ),
#     (
#         "Missing scope",
#         403,
#         _call_server("tools/list", {}, token=issue_token("alice", scopes=[])),
#     ),
#     (
#         "Tool call (valid)",
#         200,
#         _call_server(
#             "tools/call",
#             {"name": "echo", "arguments": {"message": "hello MCP"}},
#             token=_valid_token,
#         ),
#     ),
# ]

# all_passed = True
# for label, expected_status, (status, body) in _tests:
#     passed = status == expected_status
#     all_passed = all_passed and passed
#     if status == 200 and "tools" in body:
#         result = f"tools={[t['name'] for t in body['tools']]}"
#     elif status == 200 and "content" in body:
#         result = body["content"][0]["text"]
#     else:
#         result = body.get("detail") or body.get("error", "")
#     mark = "✓" if passed else f"✗ (expected {expected_status})"
#     print(f"  {mark} {label:<42} {status}  {result}")

# print(
#     f"\n{'All tests passed!' if all_passed else 'Some tests failed — check your validate_token() implementation.'}"
# )


# # %%
# """
# **Discussion:**

# 1. **The silent failure.** Remove `audience=_MCP_SERVER_URL` from your `jwt.decode()` and re-run. What changes in the test output? What would an attacker need to exploit this — and why might it be easier than it sounds in a real multi-server deployment?

# 2. **Symmetric vs. asymmetric keys.** The mock uses HMAC-SHA256 with a shared secret. Production MCP servers use RS256 (asymmetric). What's the trust-model difference — who needs access to the key material in each case? What attack becomes possible if the shared secret leaks?

# 3. **Scope granularity.** The current scope `mcp:tools` grants access to all tools. Design a finer-grained scheme for a server that exposes: read files, write files, execute shell commands, access API credentials. Which operations would you separate, and why? What's the tradeoff with UX?

# 4. **The full OAuth flow.** The `resource_metadata` URL in the 401 response is the entry point for the real MCP OAuth 2.1 handshake. Read the [MCP authorization tutorial](https://modelcontextprotocol.io/docs/tutorials/security/authorization) and trace how a client would go from that 401 response to a valid access token. What trust assumptions does each step make?
# """


# %%
"""
## Summary

Today covered three interconnected themes: the **attack surface for prompt injection**, the challenge of **AI control through monitoring**, and the **attack surface for harm** through a coding agent case study.

**Injection Attack Surface**

- **Indirect prompt injection** is the fundamental vulnerability of LLM applications — there is no equivalent of parameterized queries
- The attack surface spans every channel carrying untrusted input: user messages, RAG-retrieved documents, and MCP tool responses — all exploit the same fundamental inability to separate data from instructions

**AI Control**

- AI control is the problem of how to safely deploy AI systems that might be **intentionally subverting** your safety measures — unlike alignment (making the AI want the right things), control assumes the AI *may* be misaligned and focuses on limiting what it can get away with
- LLM-based monitors are one of the basic building block for AI control - direct human oversight won't scale. Capability gap between the monitor and monitored system matters. Trusted monitoring relies on weaker but more trusted or well understood models to supervise stronger but less trusted ones.
- **Chain-of-thought monitoring** is a uniquely powerful control lever with no analogue in human-insider-threat scenarios — but its value depends on whether CoT remains faithful in the future
- Control protocols (trusted monitoring, defer-to-trusted, resampling) formalize the **safety–usefulness tradeoff** — more aggressive auditing catches more attacks but blocks more legitimate work

**Coding Agents**
- Even a coding agent with "just" file write permissions has a **surprisingly wide attack surface**: executable code injection, credential access (env vars, config files, cloud metadata), lateral movement (git hooks, CI pipelines), persistence mechanisms, and supply chain manipulation
- **Affordances and autonomy** — not the injection vector — determine the blast radius of a successful attack
- Thinking systematically about mitigations (sandboxing, least-privilege, human-in-the-loop, monitoring) is essential because no single defense is sufficient

### Further Reading

**Application vulnerabilities**

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — A community-driven security resource identifying the most critical vulnerabilities in LLM applications, covering risks from prompt injection to insecure output handling.
- [PromptArmor, "Data Exfiltration from Slack AI via Indirect Prompt Injection"](https://promptarmor.substack.com/p/data-exfiltration-from-slack-ai-via) — Demonstrates how attackers can steal confidential data from private Slack channels by posting malicious instructions in public channels that manipulate Slack's AI to exfiltrate sensitive information.
- [Radware, "ZombieAgent"](https://www.radware.com/blog/threat-intelligence/zombieagent/) — A zero-click indirect prompt injection vulnerability targeting OpenAI's Deep Research agent, enabling attackers to implant malicious rules into an AI agent's long-term memory via hidden directives in emails or documents, with all exfiltration occurring silently within OpenAI's cloud infrastructure.
- [LayerX, "ChatGPT Tainted Memories"](https://layerxsecurity.com/blog/layerx-identifies-vulnerability-in-new-chatgpt-atlas-browser/) — Discovers a vulnerability in OpenAI's ChatGPT Atlas browser allowing CSRF attacks to inject malicious instructions into a user's ChatGPT memory, exploiting the browser's default logged-in state and lack of anti-phishing protections.
- [Trail of Bits, "Weaponizing Image Scaling Against Production AI Systems"](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems) — Describes a vulnerability where crafted images appear harmless at full resolution but contain hidden prompt injections when scaled down by AI systems, demonstrated across Google Gemini, Vertex AI, and others.

**AI Control**
- [ControlArena documentation](https://ukgovernmentbeis.github.io/control-arena/)
- [AI Control Reading List](https://blog.redwoodresearch.org/p/guide) — A curated guide to Redwood Research's work on AI control. Organizes their publications into a quick introduction and an extensive guide covering threat models, control techniques, and safety cases.

- [Ctrl-Z: Controlling AI Agents via Resampling](https://blog.redwoodresearch.org/p/ctrl-z-controlling-ai-agents-via) — Introduces "resample protocols" to prevent malicious AI agents from causing harm in multi-step environments. Evaluates safety measures against adversarial agents attempting to execute dangerous code.

- [An Overview of Control Measures](https://blog.redwoodresearch.org/p/an-overview-of-control-measures) — Outlines strategies for ensuring AI systems remain controllable even if they attempt to circumvent safeguards. Categorizes measures into software infrastructure, AI/human components, and security improvements.

- [Prioritizing Threats for AI Control](https://blog.redwoodresearch.org/p/prioritizing-threats-for-ai-control) — A framework for categorizing and prioritizing threats from misaligned AI. Argues that preventing "rogue internal deployments" (AIs running unmonitored with escalated permissions) should be the top priority.

- [An Overview of Areas of Control Work](https://blog.redwoodresearch.org/p/an-overview-of-areas-of-control-work) — Maps the major research and implementation areas for AI control: evaluation settings, model capabilities, countermeasures, and infrastructure prototyping.

- [AI Catastrophes and Rogue Deployments](https://blog.redwoodresearch.org/p/ai-catastrophes-and-rogue-deployments) — Introduces the concept of "rogue deployments" (running AI models without safety measures) as a framework for classifying AI catastrophes. Argues that launching an unauthorized deployment is far simpler than directly causing a catastrophe.

- [Recent Redwood Research Project Proposals](https://blog.redwoodresearch.org/p/recent-redwood-research-project-proposals) — A collection of empirical AI safety and security research proposals spanning AI control, training-time alignment, and model interpretability.

- [What's Worse, Spies or Schemers?](https://blog.redwoodresearch.org/p/whats-worse-spies-or-schemers) — Compares two insider threat problems: human employees who might misuse AI systems ("spies") vs. AI models that might act deceptively ("schemers"). Concludes schemers pose greater risk if they occur, though spies are more likely to emerge first.

- [Measuring No-CoT Math Time Horizon](https://blog.redwoodresearch.org/p/measuring-no-cot-math-time-horizon) — Relevant for chain-of-thought monitoring: Measures how well AI models solve math problems in a single forward pass without chain-of-thought reasoning.

**MCP protocol**
- [What is Model Context Protocol (MCP)?](https://modelcontextprotocol.io/docs/getting-started/intro) — An introduction to MCP, the emerging standard for giving LLMs structured access to tools and data.
- [MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers](https://arxiv.org/abs/2508.14925) — The first systematic benchmark for evaluating LLM agent resistance to tool poisoning attacks on MCP servers, finding attack success rates exceeding 70% across 20 prominent LLM agents when malicious instructions are embedded in tool metadata.
"""
