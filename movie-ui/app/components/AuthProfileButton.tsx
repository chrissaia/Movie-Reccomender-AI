"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

export default function AuthProfileButton() {
  const router = useRouter();
  const { user, isSignedIn } = useUser();
  const fallbackInitial =
    user?.firstName?.[0] ?? user?.username?.[0] ?? user?.primaryEmailAddress?.emailAddress?.[0] ?? "P";

  if (!isSignedIn) {
    return (
      <SignInButton mode="modal">
        <button className="auth-profile-btn">Sign In</button>
      </SignInButton>
    );
  }

  return (
    <button className="auth-profile-btn" onClick={() => router.push("/profile")}>
      {user?.imageUrl ? (
        <img
          aria-hidden="true"
          className="auth-profile-avatar"
          src={user.imageUrl}
          alt=""
        />
      ) : (
        <span aria-hidden="true" className="auth-profile-icon">
          {fallbackInitial.toUpperCase()}
        </span>
      )}
      <span>Profile</span>
    </button>
  );
}
