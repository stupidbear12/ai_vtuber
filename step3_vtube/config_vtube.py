# config_vtube.py - emeth VTube Studio 설정 (전체 리깅 버전)

# ─── API 연결 설정 ────────────────────────────────────────
VTS_HOST = "localhost"
VTS_PORT = 8001
PLUGIN_NAME = "emeth-controller"
PLUGIN_DEVELOPER = "emeth"
PLUGIN_ICON = None
TOKEN_FILE = "./vts_token.json"

# ─── Live2D 파라미터 이름 (Akari 기준, 모델에 따라 조정) ──
PARAM_FACE_ANGLE_X   = "FaceAngleX"      # 고개 좌우 (-30 ~ 30)
PARAM_FACE_ANGLE_Y   = "FaceAngleY"      # 고개 상하 (-30 ~ 30)
PARAM_FACE_ANGLE_Z   = "FaceAngleZ"      # 고개 기울임 (-30 ~ 30)
PARAM_EYE_OPEN_L     = "EyeOpenLeft"     # 왼눈 크기 (0 ~ 1)
PARAM_EYE_OPEN_R     = "EyeOpenRight"    # 오른눈 크기 (0 ~ 1)
PARAM_EYE_SMILE_L    = "EyeSmileLeft"    # 왼눈 웃음 (0 ~ 1)
PARAM_EYE_SMILE_R    = "EyeSmileRight"   # 오른눈 웃음 (0 ~ 1)
PARAM_BROW_L         = "BrowLeftY"       # 왼눈썹 높이 (-1 ~ 1)
PARAM_BROW_R         = "BrowRightY"      # 오른눈썹 높이 (-1 ~ 1)
PARAM_BROW_FORM_L    = "BrowLeftForm"    # 왼눈썹 모양 (-1 ~ 1)
PARAM_BROW_FORM_R    = "BrowRightForm"   # 오른눈썹 모양 (-1 ~ 1)
PARAM_MOUTH_OPEN     = "MouthOpenY"      # 입 벌림 (0 ~ 1)
PARAM_MOUTH_FORM     = "MouthForm"       # 입 모양 (-1 웃음 ~ 1 슬픔 반대)
PARAM_BODY_ANGLE_X   = "BodyAngleX"      # 몸 좌우 (-10 ~ 10)
PARAM_BODY_ANGLE_Y   = "BodyAngleY"      # 몸 앞뒤 (-10 ~ 10)
PARAM_BREATH         = "ParamBreath"     # 호흡 (0 ~ 1)
PARAM_HAIR_SIDE      = "HairFront"       # 앞 사이드 머리카락 (-10 ~ 10, 모델에 따라 조정)
PARAM_HAIR_BACK      = "HairBack"        # 뒷 머리카락 (-10 ~ 10, 모델에 따라 조정)

# ─── 표정 세트 (VTube Studio 핫키 이름) ──────────────────
EXPRESSIONS = {
    "default":   "default",
    "happy":     "happy",
    "calm":      "calm",
    "speaking":  "speaking",
    "emotional": "emotional",
}

# ─── 아이들 애니메이션 튜닝 ──────────────────────────────
IDLE_BREATH_PERIOD    = 4.0    # 호흡 주기 (초)
IDLE_BLINK_MIN        = 3.0    # 눈 깜빡임 최소 간격 (초)
IDLE_BLINK_MAX        = 6.0    # 눈 깜빡임 최대 간격 (초)
IDLE_BLINK_DURATION   = 0.12   # 눈 깜빡임 속도 (초)
IDLE_HEAD_AMPLITUDE   = 2.0    # 머리 흔들림 범위 (도)
IDLE_HEAD_PERIOD      = 8.0    # 머리 흔들림 주기 (초)
IDLE_BODY_AMPLITUDE   = 1.0    # 몸 흔들림 범위

# ─── 말하기 튜닝 ────────────────────────────────────────
SPEAKING_EYE_SQUINT   = 0.85   # 말할 때 눈 좁아짐 (1.0 기준)
SPEAKING_BODY_LEAN    = -2.0   # 말할 때 몸 앞으로 기울기

# ─── 성경 구절 모드 튜닝 ────────────────────────────────
BIBLE_BREATH_PERIOD   = 6.0    # 성경 읽을 때 호흡 주기 (초)
BIBLE_EYE_CLOSE       = 0.1    # 눈 거의 감음
BIBLE_HEAD_DOWN       = -5.0   # 고개 숙임 (도)
BIBLE_MOUTH_MAX       = 0.3    # 입 최대 개방량

# ─── 클로징 튜닝 ────────────────────────────────────────
CLOSING_HEAD_DOWN     = -8.0   # 클로징 고개 숙임
CLOSING_DURATION      = 2.0    # 클로징 애니메이션 시간 (초)
