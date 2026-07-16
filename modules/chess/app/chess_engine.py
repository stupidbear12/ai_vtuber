# -*- coding: utf-8 -*-
"""
체스 엔진 래퍼 — Stockfish + python-chess

시온(AI) vs 시청자 연합 체스 대국을 관리한다.
- 시온 측: Stockfish 엔진 (난이도 조절 가능)
- 시청자 측: 채팅 투표로 결정
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import chess
import chess.engine

logger = logging.getLogger("chess_engine")

# Stockfish 경로
_MODULE_DIR = Path(__file__).resolve().parent.parent
STOCKFISH_PATH = os.environ.get(
    "STOCKFISH_PATH",
    str(_MODULE_DIR / "stockfish" / "stockfish-windows-x86-64-avx2.exe"),
)


class GamePhase(str, Enum):
    IDLE = "idle"              # 대국 없음
    SION_TURN = "sion_turn"    # 시온 차례
    VOTE_OPEN = "vote_open"    # 시청자 투표 중
    VOTE_CLOSED = "vote_closed"  # 투표 마감, 수 적용 중
    GAME_OVER = "game_over"    # 대국 종료


@dataclass
class VoteResult:
    move_uci: str
    votes: int


@dataclass
class GameState:
    board: chess.Board = field(default_factory=chess.Board)
    phase: GamePhase = GamePhase.IDLE
    sion_color: chess.Color = chess.WHITE   # 시온이 백(선공)
    votes: dict[str, str] = field(default_factory=dict)  # user_id -> move_uci
    vote_deadline: float = 0.0
    move_history: list[dict] = field(default_factory=list)
    result: Optional[str] = None  # "sion_win", "viewer_win", "draw"
    skill_level: int = 5  # Stockfish 난이도 0~20
    vote_duration: int = 30  # 투표 시간(초)


class ChessEngine:
    """체스 대국 매니저."""

    def __init__(self):
        self.game = GameState()
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._vote_task: Optional[asyncio.Task] = None

    async def start_engine(self):
        """Stockfish 엔진을 시작한다."""
        if self._engine:
            return
        try:
            transport, engine = await chess.engine.popen_uci(STOCKFISH_PATH)
            self._engine = engine
            logger.info("[Chess] Stockfish 시작 완료: %s", STOCKFISH_PATH)
        except Exception as e:
            logger.error("[Chess] Stockfish 시작 실패: %s", e)
            raise

    async def stop_engine(self):
        """Stockfish 엔진을 종료한다."""
        if self._engine:
            await self._engine.quit()
            self._engine = None

    def new_game(self, sion_color: str = "white", skill_level: int = 5,
                 vote_duration: int = 30) -> dict:
        """새 대국을 시작한다."""
        if self.game.phase not in (GamePhase.IDLE, GamePhase.GAME_OVER):
            raise ValueError("이미 대국이 진행 중입니다.")

        self.game = GameState(
            sion_color=chess.WHITE if sion_color == "white" else chess.BLACK,
            skill_level=max(0, min(20, skill_level)),
            vote_duration=vote_duration,
        )
        self.game.phase = GamePhase.SION_TURN if sion_color == "white" else GamePhase.VOTE_OPEN

        if self.game.phase == GamePhase.VOTE_OPEN:
            self.game.vote_deadline = time.time() + vote_duration

        return self._state_dict()

    async def sion_move(self) -> dict:
        """시온(Stockfish)이 수를 둔다."""
        if self.game.phase != GamePhase.SION_TURN:
            raise ValueError("시온 차례가 아닙니다.")
        if not self._engine:
            await self.start_engine()

        # Stockfish 난이도 설정
        await self._engine.configure({"Skill Level": self.game.skill_level})

        # 생각 시간: 난이도에 따라 조절 (약한 수준이면 짧게)
        think_time = 0.5 + (self.game.skill_level / 20) * 2.0

        result = await self._engine.play(
            self.game.board,
            chess.engine.Limit(time=think_time),
        )
        move = result.move
        san = self.game.board.san(move)

        # 수 적용
        self.game.board.push(move)
        self.game.move_history.append({
            "side": "sion",
            "uci": move.uci(),
            "san": san,
            "fen": self.game.board.fen(),
        })

        # 게임 종료 체크
        if self.game.board.is_game_over():
            self._resolve_game()
        else:
            # 시청자 투표 시작
            self.game.phase = GamePhase.VOTE_OPEN
            self.game.votes = {}
            self.game.vote_deadline = time.time() + self.game.vote_duration

        return {
            "move_uci": move.uci(),
            "move_san": san,
            "state": self._state_dict(),
        }

    def submit_vote(self, user_id: str, move_str: str) -> dict:
        """시청자가 수를 투표한다."""
        if self.game.phase != GamePhase.VOTE_OPEN:
            return {"success": False, "reason": "투표 기간이 아닙니다."}

        # UCI 또는 SAN 형식 모두 지원
        try:
            move = self.game.board.parse_san(move_str)
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            try:
                move = chess.Move.from_uci(move_str)
                if move not in self.game.board.legal_moves:
                    return {"success": False, "reason": f"불가능한 수: {move_str}"}
            except (chess.InvalidMoveError, ValueError):
                return {"success": False, "reason": f"잘못된 형식: {move_str}"}

        self.game.votes[user_id] = move.uci()
        return {
            "success": True,
            "move_uci": move.uci(),
            "move_san": self.game.board.san(move),
            "total_votes": len(self.game.votes),
        }

    def close_vote(self) -> dict:
        """투표를 마감하고 가장 많이 득표한 수를 적용한다."""
        if self.game.phase != GamePhase.VOTE_OPEN:
            raise ValueError("투표 기간이 아닙니다.")

        if not self.game.votes:
            # 투표 없으면 랜덤 수
            import random
            legal = list(self.game.board.legal_moves)
            chosen = random.choice(legal)
            chosen_uci = chosen.uci()
            vote_count = 0
        else:
            # 최다 득표 수 선택
            tally: dict[str, int] = {}
            for move_uci in self.game.votes.values():
                tally[move_uci] = tally.get(move_uci, 0) + 1

            chosen_uci = max(tally, key=tally.get)
            vote_count = tally[chosen_uci]

        move = chess.Move.from_uci(chosen_uci)
        san = self.game.board.san(move)

        # 수 적용
        self.game.board.push(move)
        self.game.move_history.append({
            "side": "viewer",
            "uci": chosen_uci,
            "san": san,
            "fen": self.game.board.fen(),
            "votes": vote_count,
            "total_voters": len(self.game.votes),
        })
        self.game.votes = {}

        # 게임 종료 체크
        if self.game.board.is_game_over():
            self._resolve_game()
        else:
            self.game.phase = GamePhase.SION_TURN

        return {
            "move_uci": chosen_uci,
            "move_san": san,
            "vote_count": vote_count,
            "total_voters": len(self.game.move_history[-1].get("total_voters", 0)) if isinstance(self.game.move_history[-1].get("total_voters"), int) else 0,
            "state": self._state_dict(),
        }

    def get_vote_tally(self) -> dict:
        """현재 투표 현황을 반환한다."""
        tally: dict[str, int] = {}
        for move_uci in self.game.votes.values():
            tally[move_uci] = tally.get(move_uci, 0) + 1

        # SAN 변환
        results = []
        for uci, count in sorted(tally.items(), key=lambda x: -x[1]):
            try:
                san = self.game.board.san(chess.Move.from_uci(uci))
            except Exception:
                san = uci
            results.append({"uci": uci, "san": san, "votes": count})

        return {
            "total_voters": len(self.game.votes),
            "moves": results,
            "deadline": self.game.vote_deadline,
            "remaining_sec": max(0, self.game.vote_deadline - time.time()),
        }

    def get_legal_moves(self) -> list[str]:
        """현재 합법 수 목록을 SAN으로 반환한다."""
        return [self.game.board.san(m) for m in self.game.board.legal_moves]

    def resign(self, side: str = "sion") -> dict:
        """기권한다."""
        if side == "sion":
            self.game.result = "viewer_win"
        else:
            self.game.result = "sion_win"
        self.game.phase = GamePhase.GAME_OVER
        return self._state_dict()

    def _resolve_game(self):
        """게임 종료 판정."""
        self.game.phase = GamePhase.GAME_OVER
        outcome = self.game.board.outcome()
        if outcome is None:
            self.game.result = "draw"
        elif outcome.winner == self.game.sion_color:
            self.game.result = "sion_win"
        elif outcome.winner is not None:
            self.game.result = "viewer_win"
        else:
            self.game.result = "draw"

    def _state_dict(self) -> dict:
        """현재 게임 상태를 딕셔너리로 반환한다."""
        board = self.game.board

        # 마지막 수 정보
        last_move = None
        if self.game.move_history:
            last_move = self.game.move_history[-1]

        return {
            "phase": self.game.phase.value,
            "fen": board.fen(),
            "sion_color": "white" if self.game.sion_color == chess.WHITE else "black",
            "turn": "white" if board.turn == chess.WHITE else "black",
            "is_sion_turn": board.turn == self.game.sion_color,
            "move_count": len(self.game.move_history),
            "is_check": board.is_check(),
            "is_game_over": board.is_game_over(),
            "result": self.game.result,
            "last_move": last_move,
            "skill_level": self.game.skill_level,
            "vote_duration": self.game.vote_duration,
            "vote_deadline": self.game.vote_deadline if self.game.phase == GamePhase.VOTE_OPEN else None,
            "vote_remaining_sec": max(0, self.game.vote_deadline - time.time()) if self.game.phase == GamePhase.VOTE_OPEN else None,
            "legal_moves_count": len(list(board.legal_moves)),
        }
