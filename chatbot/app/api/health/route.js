import { NextResponse } from "next/server";

const OLLAMA_URL = process.env.OLLAMA_TUNNEL_URL || "http://localhost:11434";

export async function GET() {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (res.ok) {
      return NextResponse.json({ status: "ok" });
    }
    return NextResponse.json({ status: "error", detail: `HTTP ${res.status}` });
  } catch (err) {
    return NextResponse.json({ status: "error", detail: err.message });
  }
}
