"use client";

import { useState } from "react";
import QuestionRenderer from "@/components/QuestionRenderer";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/**
 * @param {{ cards: Array<{ id: string; front: string; back: string }>; heading?: string }} props
 */
export default function FlashcardStack({ cards = [], heading = "Flashcards" }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [isFalling, setIsFalling] = useState(false);
  const [isShifting, setIsShifting] = useState(false);
  const [isFlipping, setIsFlipping] = useState(false);

  const maxCards = 5;

  const getVisibleCards = () => {
    const visibleCards = [];
    for (let i = 0; i < maxCards && currentIndex + i < cards.length; i++) {
      visibleCards.push(cards[currentIndex + i]);
    }
    return visibleCards;
  };

  const handleNext = () => {
    if (currentIndex < cards.length - 1) {
      setIsFalling(true);

      setTimeout(() => {
        setIsShifting(true);
      }, 400);

      setTimeout(() => {
        setCurrentIndex((prev) => prev + 1);
        setIsFlipped(false);
        setIsFalling(false);
        setIsShifting(false);
      }, 800);
    }
  };

  const handleFlip = () => {
    setIsFlipping(true);
    setIsFlipped(!isFlipped);

    setTimeout(() => {
      setIsFlipping(false);
    }, 750);
  };

  const visibleCards = getVisibleCards();

  if (!cards.length) {
    return (
      <div
        className="container section-rule"
        style={{ paddingTop: 24, paddingBottom: 24 }}
      >
        <div className="meta-text" style={{ color: "var(--muted)" }}>
          No flashcards available.
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        backgroundColor: "var(--bg)",
        paddingTop: 24,
        paddingBottom: 32,
      }}
    >
      <div className="container" style={{ maxWidth: 900 }}>
        <div style={{ marginBottom: 24 }}>
          <h2
            style={{
              fontFamily: "var(--font-playfair)",
              fontSize: 32,
              color: "var(--ink)",
              marginBottom: 6,
            }}
          >
            {heading}
          </h2>
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            Card {currentIndex + 1} of {cards.length}
          </div>
        </div>

        <div className="relative w-full h-96">
          {visibleCards.map((card, index) => {
            const getTransform = () => {
              if (index === 0 && isFalling)
                return "translateX(0px) translateY(0px)";
              if (isShifting && index > 0) {
                return `translateX(${(index - 1) * 8}px) translateY(${(index - 1) * 8}px)`;
              }
              return `translateX(${index * 8}px) translateY(${index * 8}px)`;
            };

            return (
              <div
                key={`${card.id}-${currentIndex}-${index}`}
                className={`absolute w-full h-full transition-all duration-500 ${index === 0 && isFalling ? "animate-[fall_0.8s_ease-in-out_forwards]" : ""} ${index === 0 && isFalling ? "opacity-0" : "opacity-100"}`}
                style={{ zIndex: maxCards - index, transform: getTransform() }}
              >
                <div className="flip-card-container h-full relative">
                  <button
                    type="button"
                    className={`flip-card h-full cursor-pointer ${index === 0 && isFlipped ? "flipped" : ""}`}
                    onClick={() => {
                      if (index === 0) handleFlip();
                    }}
                    onKeyDown={(event) => {
                      if (index !== 0) return;
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        handleFlip();
                      }
                    }}
                    style={{
                      width: "100%",
                      border: 0,
                      padding: 0,
                      background: "transparent",
                    }}
                  >
                    <div className="flip-card-front">
                      <Card
                        className="w-full h-full p-0! gap-0! transition-all duration-300"
                        style={{
                          backgroundColor: "var(--card-bg)",
                          border: "0.5px solid var(--rule)",
                        }}
                      >
                        <CardContent className="p-8 w-full h-full px-8!">
                          <div className="text-center h-full flex flex-col justify-center">
                            <div
                              className="meta-text"
                              style={{ marginBottom: 16 }}
                            >
                              Question
                            </div>
                            <div
                              style={{
                                fontSize: 20,
                                lineHeight: 1.5,
                                color: "var(--ink)",
                                fontFamily: "var(--font-playfair)",
                              }}
                            >
                              <QuestionRenderer>{card.front}</QuestionRenderer>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    <div className="flip-card-back">
                      <Card
                        className="w-full h-full p-0! gap-0! transition-all duration-300"
                        style={{
                          backgroundColor: "var(--card-bg)",
                          border: "0.5px solid var(--rule)",
                        }}
                      >
                        <CardContent className="p-8 w-full h-full px-8!">
                          <div className="text-center h-full flex flex-col justify-center">
                            <div
                              className="meta-text"
                              style={{
                                marginBottom: 16,
                                color: "var(--vermillion)",
                              }}
                            >
                              Answer
                            </div>
                            <div
                              style={{
                                fontSize: 18,
                                lineHeight: 1.6,
                                color: "var(--ink)",
                              }}
                            >
                              <QuestionRenderer>{card.back}</QuestionRenderer>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  </button>

                  {index === 0 && !isFlipping && (
                    <div className="absolute top-4 right-4 z-10">
                      <Button
                        onClick={(event) => {
                          event.stopPropagation();
                          handleNext();
                        }}
                        disabled={currentIndex >= cards.length - 1 || isFalling}
                        size="sm"
                        className="btn"
                        style={{
                          opacity:
                            currentIndex >= cards.length - 1 || isFalling
                              ? 0.5
                              : 1,
                        }}
                      >
                        Next <span className="ml-2">→</span>
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
