"use client";

import DemoNotice from "@/components/DemoNotice";
import ShortBitStack from "@/components/ShortBitStack";
import { dummyShortBitSet } from "./dummydata";

export default function ShortBitPage() {
  return (
    <>
      <DemoNotice label="static short-bit sample data" />
      <ShortBitStack
        items={dummyShortBitSet.items}
        heading={dummyShortBitSet.title || "Short Bit"}
      />
    </>
  );
}
