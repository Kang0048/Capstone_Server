from flask import Flask, request, jsonify
import whisper
import tempfile
import os
from pydub import AudioSegment  # pip install pydub

app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    print(request.files)
    audio_file = request.files["audio"]

    # 안전한 임시 파일 경로 생성
    temp_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)  # 파일 핸들 닫기 (Windows 충돌 방지)
    
    converted_audio_path = temp_audio_path.replace(".wav", "_converted.wav")

    try:
        # 파일 저장
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        # === pydub을 사용하여 변환 ===
        audio = AudioSegment.from_file(temp_audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz 모노 변환
        audio.export(converted_audio_path, format="wav")

        # === Whisper로 변환된 WAV 파일 읽기 ===
        audio_data = whisper.load_audio(converted_audio_path)  # 변환된 파일 읽기
        result = model.transcribe(audio_data)  # Whisper로 텍스트 변환

        return jsonify({"text": result["text"]})

    finally:
        # 파일이 존재하면 삭제
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
