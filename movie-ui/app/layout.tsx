import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const faviconDataUrl =
  "data:image/png;base64,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";

export const metadata: Metadata = {
  title: "Movie UI",
  description: "Curate your next movie night",
  icons: {
    icon: faviconDataUrl,
    shortcut: faviconDataUrl,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider>
      <html
        lang="en"
        className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      >
        <body className="min-h-full flex flex-col">{children}</body>
      </html>
    </ClerkProvider>
  );
}
