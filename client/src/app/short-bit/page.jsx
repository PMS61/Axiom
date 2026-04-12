"use client";

import ShortBitStack from "@/components/ShortBitStack";
import { dummyShortBitSet } from "./dummydata";

export default function ShortBitPage() {
  return (
    <ShortBitStack
      items={dummyShortBitSet.items}
      heading={dummyShortBitSet.title || "Short Bit"}
    />
  );
}
