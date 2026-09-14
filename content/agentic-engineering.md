Agentic engineering follows a structured "research, plan, implement" loop. The key focus is distinguishing where a "human leans in" versus where an "agent runs it."

* 🧠: Human leans in
* ⚙️: Agent runs it

## Research

1. 🧠🔬 **Design Together**: Engage with every brainstorming decision and design section yourself. Design is never delegated to agents; delegated design is slop.
   * *When Unsure:* Seek an adversarial second opinion. A separate agent session on a different model family should judge the decision or design section. *(Note: This applies to both brainstorming decisions in Step 1 and spec review in Step 3).*
2. ⚙️📡 **Write the Spec**: Specs are written for agents, not humans. If you actively engaged in the design phase, you do not need to line-read the document for reassurance.
3. ⚙️📡 **Adversarial Spec Review**: A separate agent session reviews the spec and reports findings. Fix, re-review, and repeat until the output converges. *(This creates a revision loop between Steps 2 and 3).*

## Plan and Implement

4. ⚙️🦾 **Plan, Then Subagent-Driven Implementation**: Convert the converged spec into an implementation plan, then execute it with a fresh implementer and reviewer per task. (`writing-plans` → `subagent-driven-development`)
   * *Review Cadence:* If the plan has > 10 tasks, pause every ~5 tasks to run `roborev-fix` so reviews never pile up. If the plan has ≤ 10 tasks, closing out reviews at the end is acceptable.

## Review

5. 🧠🔍 **Close-out Gate**: Before any PR is raised, ensure every `roborev` review is closed out, and all generated specs and plans are either converted to durable documents or deleted. *(These documents should only land in the main branch for a specific reason, e.g., a follow-up PR requires them).*

## Decision Register

6. 🧠📖 **Make the Work Durable**: Durable artifacts must state their invariants directly in domain language, using stable paths, role-based runbooks, and meaningful metadata. Planning documents may point to durable artifacts, but never the reverse. Delete the planning tree post-merge: the repository must still explain itself, and the default branch alone should be sufficient to operate and recover the system.
   * **Terminal Action:** Explain the change, open the PR, and own the merge.

---
## Addendum

Domain-Driven Design (DDD), as presented by Eric Evans, shrinks the communication gap between business and technical teams. By relying on ubiquitous language and bounded contexts, DDD translates business needs directly into technical implementation, ensuring both sides speak the same language. With agents in the loop, this link matters even more: it dictates how we state our requirements to the model and how we interpret its reasoning in return.
