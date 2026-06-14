"use client";

import { useRouter } from "next/navigation";

import AuthProfileButton from "./AuthProfileButton";

type NavAction = {
  label: string;
  href: string;
};

type AppHeaderProps = {
  leading: NavAction;
  actions?: NavAction[];
};

export default function AppHeader({ leading, actions = [] }: AppHeaderProps) {
  const router = useRouter();

  return (
    <div className="top-bar">
      <button className="pill-btn" onClick={() => router.push(leading.href)}>
        {leading.label}
      </button>

      <div className="top-actions">
        {actions.map((action) => (
          <button
            className="pill-btn"
            key={`${action.label}-${action.href}`}
            onClick={() => router.push(action.href)}
          >
            {action.label}
          </button>
        ))}

        <AuthProfileButton />
      </div>
    </div>
  );
}
