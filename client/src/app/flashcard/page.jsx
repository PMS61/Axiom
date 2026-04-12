"use client";

import FlashcardStack from "@/components/FlashcardStack";
import { dummyFlashcardSet } from "./dummydata";

export default function FlashcardsPage() {
  return (
    <FlashcardStack
      cards={dummyFlashcardSet.cards}
      heading={dummyFlashcardSet.title || "Flashcards"}
    />
  );
}
