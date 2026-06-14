export type SavedMovie = {
  movie_id: number;
  title: string;
};

export type Movie = SavedMovie & {
  poster?: string;
};

export type SavedList = {
  id: string;
  name: string;
  movies: SavedMovie[];
  createdAt: string;
};

export type UserProfile = {
  user_id: string;
  email: string | null;
  name: string | null;
};

export type SharedMember = {
  user_id: string;
  role: string;
  profile: UserProfile | null;
};

export type SharedList = {
  id: string;
  name: string;
  owner_user_id: string;
  createdAt: string;
  updatedAt: string;
  members: SharedMember[];
  movies: SavedMovie[];
};

export type UnifiedList = SavedList & {
  kind: "personal" | "shared";
  sharedWith?: string[];
  members?: SharedMember[];
  owner_user_id?: string;
};

export type Friendship = {
  id: string;
  other_user_id: string;
  other_user: UserProfile | null;
  status?: "pending" | "accepted" | "blocked";
};

export type FriendsResponse = {
  friends: Friendship[];
  incoming_requests: Friendship[];
  outgoing_requests: Friendship[];
};

export type RecommendationItem = SavedMovie & {
  score?: number;
  final_score?: number;
  combined_score?: number;
  ranker_score?: number;
  support_count?: number;
  explanations?: string[];
  poster?: string;
  overview?: string;
  year?: string;
};

export type OrganizedRow = {
  type: string;
  title: string;
  pinned?: boolean;
  row_score?: number;
  items: RecommendationItem[];
};

export type CombinedRecommendationResponse = {
  movie_ids: number[];
  top_k: number;
  headline?: string;
  taste_summary?: Record<string, unknown>;
  organized_rows?: OrganizedRow[];
  recommendations?: RecommendationItem[];
};
