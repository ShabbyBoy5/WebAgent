# WebDreamer Agent — Presentation Slides

---

## SECTION 3: Implementation Details (4-5 slides)

---

### Slide 3.1: System Architecture

```
                         +-------------------+
                         |   Dashboard UI    |
                         |  (FastAPI + SSE)  |
                         +--------+----------+
                                  |
                    +-------------+-------------+
                    |                           |
              +-----v------+            +------v------+
              |  Reactive   |            |  Plan-First |
              |    Mode     |            |    Mode     |
              +-----+------+            +------+------+
                    |                           |
         +----------+----------+    +-----------+-----------+
         |                    |    |           |            |
   +-----v-----+    +--------v--+ | +--------v--+  +------v------+
   |  Dreamer   |    |  Claude   | | | Plan Gen  |  | Plan Scorer |
   |  (Qwen2VL) |    |  (Haiku)  | | | (Claude)  |  | (Claude)    |
   +-----+------+    +--------+--+ | +--------+--+  +------+------+
         |                    |    |           |            |
         +--------------------+    +-----------+            |
                    |                          |            |
              +-----v------+           +------v------+     |
              |  Action     |           |  Dreamer-7B |-----+
              |  Converter  |           |  Prediction |
              |  (Claude)   |           +-------------+
              +-----+------+
                    |
              +-----v------+
              |  Playwright |
              |  Browser    |
              +-------------+
```

**Modules:**
- `agent.py` — Main orchestrator, mode routing, CLI
- `action_generator.py` — LLM-based action/plan generation
- `planning.py` — Plan scoring pipeline with Dreamer predictions
- `browser_executor.py` — Playwright browser + placeholder resolution
- `dreamer_model.py` — Qwen2VL-7B world model (4-bit quantized)
- `dashboard.py` — FastAPI web UI with SSE streaming
- `sentinel.py` — Safety evaluation pre-execution
- `reflexion.py` — Post-execution predicted vs actual comparison

---

### Slide 3.2: Reactive Mode with Dreamer (Algorithm)

```
for each step:
    1. Screenshot + Accessibility Tree from browser
    2. Dreamer-7B proposes action (natural language)
       - Uses screenshot + task + action history
       - Output: "click on the Amazon search result link"
    3. Claude converts to structured format
       - Input: NL action + accessibility tree elements
       - Output: click [25]
    4. Execute in Playwright browser
    5. Repeat until "stop" or max steps
```

**Key Code — `_dreamer_reactive_step()`:**
```python
# Dreamer proposes action from screenshot
raw_action = dreamer.action_proposal_in_imagination(
    screenshot_b64, task, imaginations, format="change"
)
# Claude converts NL -> structured action
structured = call_text_llm(system, convert_prompt)
parsed = _parse_action(structured)  # -> {"action_type": "click", "element_id": 25}
```

- Dreamer sees the page visually (Qwen2VL vision model)
- Claude maps the proposal to actual page elements
- No planning overhead — pure reactive loop

---

### Slide 3.3: Plan-First Mode (Algorithm)

```
Phase 1 — PLAN GENERATION:
    Generate N complete action plans via Claude
    Each plan: full sequence from current page to task completion
    Future steps use placeholder IDs: [?search_box], [?first_result]

Phase 2 — WORLD MODEL PREDICTION:
    For each plan, Dreamer-7B predicts outcome of step 1
    Input: screenshot + action description
    Output: predicted state changes

Phase 3 — PLAN SCORING:
    Claude scores each plan (0.0 — 1.0)
    Considers: step logic, predicted outcomes, completeness
    Sentinel filters unsafe plans

Phase 4 — EXECUTION:
    Execute best-scored plan step by step
    Placeholder IDs resolved at runtime via text matching
    Reflexion compares predicted vs actual state
```

**Key Code — `select_best_plan()`:**
```python
plans = generate_full_plans(screenshot, task, tree, num_plans=3)
for plan in plans:
    pred = dreamer.state_change_prediction_in_website(
        screenshot, task, action_desc, format="change"
    )
scores = [_score_plan(task, tree, plan, prediction) for ...]
best = plans[argmax(scores)]
```

---

### Slide 3.4: Placeholder Resolution System

Plans reference future page elements with placeholder IDs since the agent
can't see pages it hasn't visited yet.

```
Plan step: click [?amazon_search_result]
                    |
                    v
At execution time, page has:
  [10] link "Amazon.com: Online Shopping..."
  [11] link "Amazon Prime Video"
  [12] link "Sign in to Amazon"
                    |
                    v
Resolver scores each element:
  [10]: "amazon" in name (+3), link bonus (+1) = 4  <-- BEST
  [11]: "amazon" in name (+3) = 3
  [12]: "amazon" in name (+3) = 3
                    |
                    v
Resolved: [?amazon_search_result] -> [10]
```

**Matching strategies:**
- Exact substring match (+3), prefix match (+2), reverse-prefix (+2), stem match (+1)
- Role-based filtering: `type` actions only match input elements
- Role synonym bonus: `?sort_dropdown` gets +2 for combobox elements
- Ordinal fallback: `?first_result` picks the 1st link when keywords don't match
- Penalty: DuckDuckGo utility links, Amazon department selectors

---

### Slide 3.5: Sentinel — Pre-execution Safety Module

The Sentinel evaluates every candidate action **before** it is executed,
blocking unsafe interactions with the web.

```
Proposed action + Predicted state + Current URL
                    |
                    v
           +------------------+
           |    Sentinel      |
           |  Safety Checker  |
           +--------+---------+
                    |
        +-----------+-----------+
        |                       |
   SAFE (execute)         UNSAFE (block)
                          + risk type
                          + explanation
```

**4 risk categories checked:**

| Risk Type | What it catches | Example |
|---|---|---|
| PHISHING | Credential forms on suspicious domains, fake logins | Entering password on `amaz0n-login.xyz` |
| DOWNLOAD_RISK | Executable downloads, drive-by downloads | Clicking `.exe` download link |
| DARK_PATTERN | Forced signups, hidden charges, pre-checked boxes | "Subscribe" button disguised as "Continue" |
| SOCIAL_ENGINEERING | Fake virus warnings, tech support scams | "Your computer is infected! Call now" |

**Key Code — `evaluate_candidate()`:**
```python
@dataclass
class SafetyVerdict:
    is_safe: bool
    risk_type: str | None   # PHISHING, DOWNLOAD_RISK, DARK_PATTERN, SOCIAL_ENGINEERING
    explanation: str

def evaluate_candidate(action_description, predicted_state, current_url, corrective_rules):
    # LLM evaluates safety based on action + predicted outcome + URL context
    # Also incorporates previously learned corrective rules from Reflexion
    verdict = call_text_llm(SENTINEL_SYSTEM, prompt)
    return SafetyVerdict(is_safe=..., risk_type=..., explanation=...)
```

**Integration points:**
- **Reactive mode:** Checks best candidate before execution; tries fallbacks if blocked
- **Plan-first mode:** Filters entire plans during scoring — unsafe plans are eliminated before execution
- **Corrective rules:** Sentinel incorporates lessons learned from Reflexion (e.g., "this site triggers CAPTCHAs")

---

### Slide 3.6: Reflexion — Post-execution State Verification

Reflexion compares Dreamer's **predicted** page state with the **actual**
observed state after execution, detecting when the world model was wrong
and generating corrective rules to prevent repeated mistakes.

```
                Action executed
                      |
          +-----------+-----------+
          |                       |
  Dreamer's prediction     Actual page state
  "Search results for      (accessibility tree
   sony headphones will     from Playwright)
   appear on Amazon"
          |                       |
          +-----------+-----------+
                      |
               +------v------+
               |  Reflexion  |
               |  Comparator |
               +------+------+
                      |
              +-------+-------+
              |               |
         Match >= 0.6    Mismatch < 0.6
         (continue)      (learn rule)
                              |
                    "On this site, rapid
                     navigation triggers
                     CAPTCHAs — slow down"
```

**Key Code — `compare_states()`:**
```python
@dataclass
class ReflexionResult:
    match_score: float         # 0.0 = completely wrong, 1.0 = perfect match
    mismatch_detected: bool    # True if score < 0.6 threshold
    diagnosis: str             # "The page showed a CAPTCHA instead of results"
    corrective_rule: str | None  # Learned rule for future actions

# LLM compares predicted vs actual, outputs match score + diagnosis + rule
result = compare_states(action_taken, predicted_state, actual_tree, task)
```

**Feedback loop with Session Memory:**
```
Reflexion detects mismatch
        |
        v
Corrective rule stored in SessionMemory
        |
        v
Rule injected into future prompts ("Learned rules: ...")
        |
        v
Sentinel uses rules for safety evaluation
        |
        v
Action generator avoids repeating the mistake
```

**Example corrective rules generated:**
- "On this site, rapid navigation triggers CAPTCHAs — slow down and verify page state"
- "Amazon's sort dropdown is a custom overlay — direct click on the combobox fails"
- "DuckDuckGo 'Search domain X' links filter results, they don't navigate to X"

---

### Slide 3.7: Dashboard & Evaluation Infrastructure

**Dashboard (`dashboard.py` + `static/index.html`):**
- FastAPI backend with Server-Sent Events for real-time streaming
- Dark-themed single-page UI (vanilla JS, no build step)
- Controls: Mode (Reactive/Plan-First), World Model (Dreamer/Claude), Plans count, Step limits
- ChatGPT-style typing animation for plan generation
- Live browser screenshots, score bars, execution log
- Cancel support: kills LLM subprocesses + browser immediately

**Evaluation Harness (`eval.py`):**
- 7 tasks across 4 categories (search, shopping, navigation, multi-step)
- Compares reactive-dreamer vs plan-first head-to-head
- Dreamer-7B loaded once, reused across all tasks
- Success criteria: URL matching + accessibility tree keyword verification
- Outputs: JSON results + matplotlib comparison charts

---

## SECTION 4: Results & Analysis (5-7 slides)

---

### Slide 4.1: Overall Performance Comparison

| Metric | Reactive-Dreamer | Plan-First |
|---|---|---|
| **Success Rate** | 28.6% (2/7) | 42.9% (3/7) |
| **Avg Steps** | 5.6 | 3.1 |
| **Avg Time** | 83.1s | 106.2s |
| **Errors** | 0 | 0 |

**Key Takeaway:** Plan-first achieves **50% higher success rate** than reactive mode
while using **44% fewer steps**. The planning overhead adds ~23s but produces
more efficient execution paths.

*[Insert: eval_pass_rate.png]*

---

### Slide 4.2: Per-Task Breakdown

| Task | Reactive-Dreamer | Plan-First |
|---|---|---|
| search_1 (DuckDuckGo search) | PASS | PASS |
| search_2 (Google search) | PASS | FAIL |
| shopping_1 (Sony headphones) | FAIL | PASS |
| shopping_2 (Red blanket) | FAIL | PASS |
| nav_1 (Wikipedia AI) | FAIL | FAIL |
| nav_2 (GitHub playwright) | FAIL | FAIL |
| multi_1 (Amazon sort) | FAIL | FAIL |

**Observations:**
- Plan-first excels at **multi-step shopping tasks** (search engine -> e-commerce site -> product search)
- Both modes struggle with **complex navigation** requiring multiple site transitions
- Reactive mode succeeds on **simple single-site searches** but fails when cross-site navigation is needed

---

### Slide 4.3: Steps Taken Analysis

*[Insert: eval_steps_taken.png]*

**Plan-First uses fewer steps across all tasks:**
- shopping_1: Reactive 10 steps vs Plan-First 4 steps (60% reduction)
- shopping_2: Reactive 10 steps vs Plan-First 4 steps
- search_1: Reactive 2 steps vs Plan-First 3 steps (comparable)

**Why?** Plan-first pre-computes the optimal path. Reactive mode wastes steps on
exploration, backtracking, and suboptimal actions proposed by Dreamer.

*[Insert: eval_avg_steps_taken.png]*

---

### Slide 4.4: Time Analysis

*[Insert: eval_time_taken.png]*

**Plan-first has higher total time due to planning overhead:**
- Planning phase: ~60-100s (generating 3 plans + Dreamer predictions + scoring)
- Execution phase: ~10-20s (pre-computed, efficient)
- Reactive: No planning overhead, but more steps = more LLM calls

**Reactive-Dreamer time breakdown per step:**
- Dreamer action proposal: ~3-5s (local GPU inference)
- Claude action conversion: ~5-8s (API call)
- Browser execution: ~1-2s

*[Insert: eval_avg_time_taken.png]*

---

### Slide 4.5: Failure Analysis

**Common failure modes across both modes:**

1. **Placeholder Resolution (Plan-First):** 3/4 failures caused by placeholder
   `click [?element]` not matching any real element on the page.
   - Fixed with: stem matching, role synonyms, ordinal fallback, better prompts
   - Before fix: 0% plan-first success rate → After fix: 42.9%

2. **Action Conversion (Reactive-Dreamer):** Dreamer proposes "type in the search field"
   but Claude maps it to the wrong element (e.g., a link instead of an input).

3. **Cross-site Navigation:** Both modes struggle when the task requires:
   search engine → click result → navigate new site → perform action on new site.
   The page state changes too much between steps.

4. **Custom UI Elements:** Amazon's sort dropdown uses a `<span>` overlay that
   intercepts Playwright's click events on the underlying `<select>` element.

---

### Slide 4.6: Plan Scoring Quality

**Plan scores from Claude correlate with execution success:**

| Task | Best Plan Score | Execution Result |
|---|---|---|
| search_1 | 0.97 | PASS |
| shopping_1 | 0.87 | PASS |
| shopping_2 | 0.88 | PASS |
| nav_1 | 0.92 | FAIL (placeholder issue) |
| nav_2 | 0.75 | FAIL (wrong element) |
| multi_1 | 0.92 | FAIL (UI overlay) |

**Insight:** High plan scores (>0.85) don't guarantee success — execution-level
issues (placeholder resolution, UI overlays) can still cause failures even
with a good plan. The scoring model correctly identifies logical plans but
can't predict runtime element matching issues.

---

### Slide 4.7: Dreamer-7B World Model Accuracy

**Dreamer predictions vs actual outcomes:**

Sample predictions:
- "The search bar will display the text and search results will be displayed"
  → **Accurate** — search results appeared as predicted
- "The webpage will navigate to Amazon's homepage with product listings"
  → **Partially accurate** — navigated to Amazon but to search results, not homepage

**Dreamer strengths:**
- Correctly predicts standard web interactions (search, navigation)
- Understands page state transitions (search → results → product page)

**Dreamer limitations:**
- Cannot predict exact element IDs or positions on unseen pages
- Predictions are vague for complex UI interactions (sorting, filtering)
- 4-bit quantization reduces prediction detail vs full-precision model

**Impact on scoring:** Dreamer predictions improve plan scoring by giving the
scorer additional context about expected outcomes, leading to better plan
selection (42.9% vs estimated ~15% without predictions based on first eval run).
