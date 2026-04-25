# agentry project ideas

A running list of small, personally-useful agent projects to build with LangGraph.
Most start as two-agent hand-offs; some will grow more complex over time.
Everything here is something I'd actually use.

Status legend: `[ ]` idea · `[~]` building · `[x]` shipped

---

## [~] 1. Inbox Triager + Drafter
**Folder:** [`emails/`](./emails/)

- **Agent A (Triager):** Reads unread Gmail, classifies (reply-needed / FYI / newsletter / spam-ish), extracts action items.
- **Agent B (Drafter):** For "reply-needed", drafts a response in your voice and saves it as a Gmail draft (never sends).
- **Why it's good:** You'll use it every morning. Clear hand-off between agents. Safe (drafts only).

---

## [ ] 2. Recipe Planner + Grocery List Builder
- **Agent A (Planner):** Takes "what's in my fridge" + dietary prefs + days to plan, picks recipes.
- **Agent B (Shopper):** Diffs recipe ingredients vs. pantry, produces a categorized grocery list (produce / dairy / etc.), optionally pushes to Reminders/Todoist.
- **Why it's good:** Real weekly utility, simple state, fun to extend.

---

## [ ] 3. Personal Finance Categorizer + Reporter
- **Agent A (Categorizer):** Ingests CSV bank export, categorizes transactions (rules + LLM for ambiguous ones).
- **Agent B (Reporter):** Generates a monthly markdown report — top categories, anomalies, "you spent 3x more on takeout this month."
- **Why it's good:** Tangible insight every month. Easy to verify quality.

---

## [ ] 4. Reading List Curator + Summarizer
- **Agent A (Curator):** Watches a folder of saved URLs (or Pocket/Readwise), filters out fluff, ranks by relevance to your interests.
- **Agent B (Summarizer):** Fetches top N, produces a "morning brief" with TL;DRs and key quotes.
- **Why it's good:** Replaces doomscrolling with curated reading.

---

## [ ] 5. Home Maintenance Tracker + Scheduler
- **Agent A (Logger):** You text/voice it ("changed AC filter today"), it logs to a structured store.
- **Agent B (Scheduler):** Knows recurrence intervals, surfaces "this week: rotate mattress, replace smoke detector battery."
- **Why it's good:** Solves a real problem (forgetting maintenance). Tiny state, big value.

---

## [ ] 6. Job/Project Application Researcher + Tailor
- **Agent A (Researcher):** Given a job posting URL, scrapes role + company, pulls recent news/values.
- **Agent B (Tailor):** Rewrites your resume bullets and drafts a cover letter tuned to that role.
- **Why it's useful:** Even passive job-hunting benefit. Reusable for grant/conference apps too.

---

## [ ] 7. Travel Planner + Local Expert
- **Agent A (Planner):** Takes "5 days in Lisbon, mid-budget, like food + walking", builds a day-by-day skeleton.
- **Agent B (Local Expert):** Uses search to fill in actual restaurants/sights with hours/booking notes.
- **Why it's good:** You'll genuinely use it next time you travel.
