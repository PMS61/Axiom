"use client";

import DemoNotice from "@/components/DemoNotice";
import FlashcardStack from "@/components/FlashcardStack";
import { dummyFlashcardSet } from "./dummydata";

export default function FlashcardsPage() {
  return (
    <>
      <DemoNotice label="static flashcard sample data" />
      <FlashcardStack
        cards={dummyFlashcardSet.cards}
        heading={dummyFlashcardSet.title || "Flashcards"}
      />
    </>
  );
}
