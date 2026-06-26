import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BehaveGuard — Behavioral Pattern Test",
  description:
    "A short typing and mouse-movement test used to study behavioral biometrics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full bg-bg text-text antialiased">{children}</body>
    </html>
  );
}
