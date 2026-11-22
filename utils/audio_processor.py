import os

try:
    import static_ffmpeg
    static_ffmpeg.add_paths(weak=True)
except Exception:
    pass

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def download_youtube_audio(url :str) ->str:
    url = url.strip().strip('"').strip("'")
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"]
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        },
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        base = os.path.splitext(raw_filename)[0]
        filename = f"{base}.wav"
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    input_path = input_path.strip().strip('"').strip("'")
    base = os.path.splitext(input_path)[0]
    output_path = f"{base}_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    wav_path = wav_path.strip().strip('"').strip("'")
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []
    base = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


def cleanup_temp_files(chunks: list = None, source: str = None):
    """
    Safely deletes all temporary .wav chunks and intermediate downloaded/converted files.
    Preserves original user files located outside the temporary downloads directory.
    """
    # 1. Delete all generated audio chunk files
    if chunks:
        for chunk in chunks:
            if chunk and os.path.exists(chunk):
                try:
                    os.remove(chunk)
                    print(f"🧹 Deleted temporary chunk: {os.path.basename(chunk)}")
                except Exception as e:
                    print(f"Warning: could not delete {chunk}: {e}")

    # 2. Delete any intermediate _converted.wav or downloaded audio files
    if source:
        clean_source = source.strip().strip('"').strip("'")
        base = os.path.splitext(clean_source)[0]
        converted = f"{base}_converted.wav"
        if os.path.exists(converted) and converted != clean_source:
            try:
                os.remove(converted)
                print(f"🧹 Deleted converted WAV file: {os.path.basename(converted)}")
            except Exception:
                pass

        # If it was downloaded from YouTube or uploaded into DOWNLOAD_DIR
        abs_download_dir = os.path.abspath(DOWNLOAD_DIR)
        if os.path.exists(clean_source) and os.path.abspath(clean_source).startswith(abs_download_dir):
            try:
                os.remove(clean_source)
                print(f"🧹 Deleted temporary source file: {os.path.basename(clean_source)}")
            except Exception:
                pass



