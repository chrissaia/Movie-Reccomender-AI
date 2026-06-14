"use client";

import { SignInButton, useUser } from "@clerk/nextjs";
import { useRouter } from "next/navigation";

export default function AuthProfileButton() {
  const router = useRouter();
  const { user, isSignedIn } = useUser();

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
        <img className="auth-profile-avatar" src={user.imageUrl} alt="Profile" />
      ) : (
        <span className="auth-profile-icon">👤</span>
      )}
      <span>Profile</span>
    </button>
  );
}