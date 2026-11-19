import { verify } from "jsonwebtoken";
import { type NextRequest, NextResponse } from "next/server";

const guestPages = new Set([
  "/dashboard/demo",
  "/dashboard/demo/roadmap",
  "/dashboard/demo/discover",
  "/dashboard/demo/study",
]);

export function proxy(request: NextRequest) {
  if (guestPages.has(request.nextUrl.pathname)) return NextResponse.next();

  const token = request.cookies.get("token")?.value;

  if (!token || !process.env.JWT_SECRET) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  try {
    verify(token, process.env.JWT_SECRET);
    return NextResponse.next();
  } catch {
    return NextResponse.redirect(new URL("/login", request.url));
  }
}

export const config = {
  matcher: "/dashboard/:path*",
};
