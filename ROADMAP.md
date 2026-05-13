# 🎬 AI Movie Recommendation Platform — Current Roadmap

## What is already done ✅
- FastAPI backend is working
- TypeScript / Next frontend is working
- Movie search is working
- Single-movie recommendation flow exists
- Combined recommendation flow exists
- Cosine retrieval pipeline exists
- Ranker exists
- Inference layer exists for combined recommendations
- UI can already show recommendation results

---

# 🚦 Current Build Status

## Foundation
Search movies  
→ select favorites  
→ get recommendations  
→ view combined results

## Recommendation Engine
Cosine retrieval  
→ ranker  
→ combined inference  
→ frontend results rendering

---

# 🧭 Product Roadmap

## PHASE 1 — ACCOUNTS + SAVED MOVIES
### Goal
Let users sign in and save their movie taste.

### Scope
Sign in  
→ save favorite movies  
→ create named lists  
→ organize taste into personal collections

### Features
- Sign in / sign out
- Persistent user profile
- Save favorite movies
- Create custom lists
- Add movies to lists
- Rename lists
- Remove movies from lists

### Example lists
- `Drama with friends`
- `Mind-bending sci fi`
- `Movies that changed me`
- `Rainy day movies`

### Suggested tables
- `users`
- `favorite_movies`
- `lists`
- `list_movies`

### Next step →
**Add Sign In button to the UI and wire basic auth state**

---

## PHASE 2 — SOCIAL LISTS
### Goal
Let signed-in users connect with other signed-in users.

### Scope
Add friends  
→ share lists  
→ combine favorite movie lists  
→ discover overlap together

### Features
- Add friends
- View friend profiles
- Share movie lists
- Invite collaborators to lists
- Combine favorite lists between users
- See mutual movie taste

### Example outputs
- `You and Sam both love slow-burn thrillers`
- `Your shared favorites lean toward crime dramas and thoughtful sci fi`

### Suggested tables
- `friends`
- `list_collaborators`
- `shared_lists`

### Next step →
**After personal lists work, add friend relationships**

---

## PHASE 3 — COMMONALITY DETECTION
### Goal
Explain what multiple movies have in common.

### Scope
Find strong commonalities  
→ summarize shared taste  
→ generate recommendation buckets from those commonalities

### Example outputs
- `Because you like the genre drama...`
- `Because your picks share psychological tension and dark tone...`
- `Because you often choose character-driven thrillers...`

### Signals
- genre
- themes
- tone
- pacing
- era
- mood
- director
- actor overlap
- keyword overlap

### Next step →
**Generate taste summaries from selected favorites and saved lists**

---

## PHASE 4 — NATURAL LANGUAGE FILTERING
### Goal
Turn user prompts into recommendation filters.

### Scope
User prompt  
→ parse constraints  
→ convert to filters  
→ rank the best matches

### Example prompts
- `I want a dark sci fi movie from the 90s with a strong female lead`
- `Give me a disturbing psychological thriller with snow vibes`
- `I want something like The Shining but less supernatural`

### Features
- prompt parsing
- filter extraction
- recommendation constraints
- semantic search layer

### Next step →
**Support prompt-to-filter recommendations after taste summaries are stable**

---

## PHASE 5 — EXPLAINABLE RECOMMENDATIONS
### Goal
Show why each recommendation was chosen.

### Scope
Recommendation selected  
→ surface strongest signals  
→ explain in plain English

### Example outputs
- `Recommended because it shares a director, tone, and era with The Shining`
- `Recommended because it overlaps with your preference for dark crime dramas`
- `Recommended because it matches your sci fi + mystery taste profile`

### Features
- per-movie explanation text
- list-level explanation text
- shared-profile explanation text

### Next step →
**Improve explanation quality after prompt filtering is in place**

---

## PHASE 6 — COLD-START CONVERSATIONAL SEARCH
### Goal
Help users with little or no movie history.

### Scope
Ask a few smart questions  
→ build a temporary taste profile  
→ generate recommendations

### Example flow
User has no saved movies  
→ AI asks for 3 favorites  
→ asks about tone / genre / pacing  
→ generates starter recommendations

### Features
- conversational onboarding
- taste seeding
- zero-history recommendation flow

### Next step →
**Use this after auth + lists + prompt filtering are established**

---

## PHASE 7 — TOP 5 DIFFERENCE SUMMARIES
### Goal
Explain why the top recommendations are different from one another.

### Scope
Top 5 results  
→ compare them  
→ summarize differences in tone, pace, theme, or intensity

### Example outputs
- `Movie A is more emotional and reflective`
- `Movie B leans more action-heavy and suspenseful`
- `Movie C is the darkest and most psychological of the group`

### Features
- recommendation comparison summaries
- top-5 differentiation
- user-facing ranking reasoning

### Next step →
**Add after recommendation explanations are reliable**

---

# 🛠 Current Priority Order

## Right now
Sign in  
→ save favorites  
→ create named lists

## Then
Add friends  
→ combine favorite lists  
→ detect commonalities

## Then
Prompt filters  
→ recommendation explanations  
→ cold-start conversation  
→ top-5 comparison summaries

---

# ✅ Recommended Immediate Next Step

## Build now
**Sign In button in the UI**

Then immediately after:
Save favorites  
→ create personal named lists

---

# 🔭 Near-Term End State

User signs in  
→ saves favorite movies  
→ creates lists like `Drama with friends`  
→ adds friends  
→ combines movie lists  
→ gets recommendations based on shared taste  
→ sees explanations for why each movie was chosen

---

# One-line roadmap view

Sign in  
→ Save favorites  
→ Create named lists  
→ Add friends  
→ Combine lists  
→ Detect commonalities  
→ Prompt-based filters  
→ Explain recommendations  
→ Cold-start conversation  
→ Summarize top 5 differences