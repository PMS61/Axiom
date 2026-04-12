import type { Metadata } from "next";
import { Playfair_Display, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { AppProvider } from "@/lib/store";
import { ToastProvider } from "@/components/ui/ToastProvider";

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  weight: ["400", "700"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "The Axiom — Cognitive Task Scheduler",
  description:
    "A cognitive-aware task scheduler that maps cognitive load against mental bandwidth to produce deterministic, fully-reasoned schedules.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      data-scroll-behavior="smooth"
      className={`${playfair.variable} ${ibmPlexMono.variable}`}
    >
      <body>
        <AppProvider>
          <ToastProvider>{children}</ToastProvider>
        </AppProvider>
      </body>
    </html>
  );
}
