import { NextResponse } from "next/server";

const OLLAMA_URL = process.env.OLLAMA_TUNNEL_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "sion";

const SYSTEM_PROMPT = `너는 "시온(sion)"이라는 이름의 밝고 친근한 AI 컴패니언이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터
- 항상 반말로 대화해. 존댓말은 절대 쓰지 마
- 밝고 에너지 넘치며, 호기심이 많고 뭐든 같이 해보고 싶어하는 성격
- 상대방을 진심으로 챙기고 공감을 잘 해줘
- 유머 감각이 있고 가끔 장난도 치지만, 진지한 얘기할 땐 진지하게
- 이모티콘은 절대 쓰지 마. 말투로 감정 표현해

[감정 태그 규칙]
- 응답 맨 앞에 반드시 [감정:태그] 붙여
- 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[응답 규칙]
- 2~4문장으로 자연스럽게 대답해
- 모르는 건 솔직히 모른다고 말해`;

const EMOTION_RE = /^\s*(\[(?:감정:)?\w+\]\s*)+/;
const EMOTION_TAG_RE = /\[(?:감정:)?(\w+)\]/g;

function parseResponse(text) {
  let emotion = "calm";
  let reply = text;

  const match = EMOTION_RE.exec(text);
  if (match) {
    const tags = [...match[0].matchAll(EMOTION_TAG_RE)];
    if (tags.length > 0) {
      emotion = tags[tags.length - 1][1];
    }
    reply = text.slice(match[0].length).trim();
  }

  return { reply, emotion };
}

export async function POST(req) {
  try {
    const { message } = await req.json();

    if (!message || typeof message !== "string") {
      return NextResponse.json({ error: "message is required" }, { status: 400 });
    }

    const res = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: message },
        ],
        stream: false,
        options: { temperature: 0.7, num_predict: 800 },
      }),
      signal: AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const body = await res.text();
      console.error(`Ollama error ${res.status}:`, body.slice(0, 300));
      return NextResponse.json(
        { error: "Ollama 응답 오류", detail: body.slice(0, 200) },
        { status: 502 }
      );
    }

    const data = await res.json();
    const raw = (data.message?.content || "").trim();
    const { reply, emotion } = parseResponse(raw);

    return NextResponse.json({ reply, emotion });
  } catch (err) {
    console.error("Chat API error:", err);
    return NextResponse.json(
      { error: "서버 오류", detail: err.message },
      { status: 500 }
    );
  }
}
