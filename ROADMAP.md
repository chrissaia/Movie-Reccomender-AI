# 🎬 AI Movie Recommendation Platform — Current Roadmap

## What is already done ✅

- FastAPI backend is working
- TypeScript / Next.js frontend is working
- Movie search is working
- Single-movie recommendation flow exists
- Combined recommendation flow exists
- Cosine retrieval pipeline exists
- Ranker exists
- Inference layer exists for combined recommendations
- UI can already show recommendation results
- Clerk sign-in UI exists
- Saved movie lists are moving from localStorage to backend persistence
- Edit Lists page exists
- My Lists page exists
- Results page exists
- Multi-row recommendation layout exists
- Backend now returns organized_rows
- Recommendation rows can now be organized by product logic, not just raw model score

---

# 🚦 Current Build Status

## Foundation

Search movies  
→ select favorites  
→ get recommendations  
→ view organized results  
→ save lists  
→ edit lists  
→ rerun recommendations from saved lists

## Recommendation Engine

Cosine retrieval  
→ candidate pool  
→ ranker  
→ combined inference  
→ taste summary  
→ organized recommendation rows  
→ frontend results rendering

---

# 🧠 Core Product Idea

The app should not simply give users a long list of movies.

The app should help users make a faster decision by organizing recommendations into useful rows.

The goal is:

- Minimize browsing time
- Maximize confidence
- Increase engagement
- Explain why each row exists

The recommendation system should rank both:

1. Movies
2. Recommendation rows

A movie can be a strong recommendation, but the page also needs to decide which rows deserve the user's attention first.

The product should feel like:

- This app understands my taste
- This app organizes my options
- This app helps me choose faster

---

# 🧭 Product Roadmap

# PHASE 1 — ACCOUNTS + SAVED MOVIES

## Goal

Let users sign in and save their movie taste.

## Scope

Sign in  
→ save favorite movies  
→ create named lists  
→ organize taste into personal collections  
→ return later and reload saved taste

## Features

- Sign in / sign out
- Persistent user profile
- Save favorite movies
- Create custom lists
- Add movies to lists
- Rename lists
- Remove movies from lists
- Delete lists
- Rerun recommendations from a saved list

## Example lists

- Drama with friends
- Mind-bending sci fi
- Movies that changed me
- Rainy day movies
- Date night movies
- Christopher Nolan favorites

## Current implementation direction

- Clerk handles frontend auth state
- Backend uses Clerk user ID through request headers for now
- Saved lists should persist in the backend database
- selectedMovies can temporarily remain in localStorage as page-to-page handoff
- Saved lists should not rely on browser localStorage

## Suggested tables

### user_profiles

- user_id
- email
- name
- created_at
- updated_at

### user_lists

- id
- user_id
- name
- created_at

### user_list_movies

- id
- list_id
- movie_id
- title
- position

## Next step

Finish backend-backed saved lists and make sure this flow works:

Sign in  
→ search movies  
→ select favorites  
→ get results  
→ save list  
→ go to My Lists  
→ refresh page  
→ list is still there

---

# PHASE 2 — ORGANIZED RECOMMENDATION ROWS

## Goal

Turn recommendations into a structured decision experience.

The app should not show one giant recommendation dump.

It should organize results into rows that help users choose faster.

## Always show top rows

These should always appear near the top:

1. Top Picks For You
2. Familiar, But Not Obvious
3. Hidden Gems You Might Like

## Dynamic rows

After the pinned rows, the app should naturally mix in rows based on the user's preferences.

### Examples

- Because You Like Action
- Because You Like Funny Superheroes
- Because You Like Psychological Tension
- Because You Liked Whiplash
- Because You Liked The Dark Knight
- Because You Like Christopher Nolan's Darker Style

## Important principle

Taste profile rows should not always be every other row.

They should be ranked naturally based on the user's actual selected movies.

The page should decide which rows matter most.

## Row types

- top_picks
- familiar_but_not_obvious
- hidden_gems
- taste_profile
- source_movie
- shared_taste
- watch_history_based

## Row ranking logic

Each possible row should receive a row_score.

### Example formula

```text
row_score =
    taste_signal_strength
  + average_movie_score
  + specificity
  + support_count
  + diversity_value
```

## Meaning of each signal

### taste_signal_strength

How strongly the row explains the user's overall taste.

Example:

If the user selects Avengers, Iron Man, Deadpool, and The Dark Knight, then "Because You Like Superhero Action" should rank highly.

### average_movie_score

How strong the movies inside that row are.

A row should not rank high if the actual movie recommendations inside it are weak.

### specificity

Specific rows should beat generic rows when they are useful.

Better:

- Because You Like Funny Superheroes

Weaker:

- Because You Like Action

### support_count

How many selected movies support the row.

A row supported by four selected movies should beat a row supported by one selected movie.

### diversity_value

Prevents the page from feeling repetitive.

Avoid:

- Because You Like Action
- Because You Like Superheroes
- Because You Like Marvel
- Because You Like Explosions

Better:

- Because You Like Superhero Action
- Because You Like Funny Superheroes
- Because You Liked The Dark Knight
- Because You Like Darker Comic Book Movies

## Current backend output

The backend now returns:

- organized_rows

This should become the main frontend rendering source.

## Next step

Update the Results page to render organized_rows directly.

---

# PHASE 3 — USER PROFILE PAGE

## Goal

Give each user a personal home base for their movie taste.

## Scope

User profile  
→ onboarding preferences  
→ saved lists  
→ friends  
→ watch history  
→ taste profile summary

## Features

- View user profile
- Edit display name
- View saved lists
- View friends
- View recent activity
- View taste summary
- View onboarding preferences
- View favorite genres
- View favorite actors/directors
- View preferred moods and pacing
- View watch history
- View ratings history

## Example profile sections

- My Taste Profile
- My Lists
- My Friends
- Recently Watched
- My Ratings
- Onboarding Preferences
- Favorite Genres
- Favorite Directors
- Favorite Actors

## Future onboarding preferences

Eventually, users should be able to answer onboarding questions like:

- What genres do you usually like?
- Do you prefer serious or fun movies?
- Do you like slow-burn movies?
- Do you like dark movies?
- Do you care more about plot, characters, visuals, or mood?
- Do you usually watch alone, with friends, or with family?
- What are three movies you love?
- What are three movies you hate?

## Suggested tables

### user_profiles

- user_id
- email
- name
- bio
- avatar_url
- created_at
- updated_at

### user_onboarding_preferences

- id
- user_id
- favorite_genres
- disliked_genres
- preferred_moods
- preferred_pacing
- preferred_decades
- favorite_movies
- disliked_movies
- created_at
- updated_at

## Next step

Create /profile page after saved lists are stable.

---

# PHASE 4 — SOCIAL LISTS + FRIENDS

## Goal

Let signed-in users connect with other signed-in users.

## Scope

Add friends  
→ view friend profiles  
→ share lists  
→ combine favorite movie lists  
→ discover overlap together

## Features

- Search users by name or email
- Send friend requests
- Accept friend requests
- Reject friend requests
- Remove friends
- View friend profiles
- View friend lists if shared
- Share movie lists
- Invite collaborators to lists
- Combine favorite lists between users
- See mutual movie taste

## Example outputs

- You and Sam both love slow-burn thrillers
- Your shared favorites lean toward crime dramas and thoughtful sci fi
- You both like dark, character-driven stories
- You both like funny action movies with ensemble casts