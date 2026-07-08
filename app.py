from flask import Flask, request, jsonify, send_file, Response, render_template
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
import uuid
import threading
import json
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from fer import FER
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
import logging
import gc
import traceback
import warnings
import io
import base64
import shutil
from datetime import datetime, timezone


# Load environment variables (kept for any future use, not required)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# In-Memory Job Store (replaces Supabase entirely)
# ---------------------------------------------------------------------------
_job_store: dict = {}
_job_store_lock = threading.Lock()


def save_job(job_id: str, **fields):
    """Create or update a job record in the in-memory store."""
    with _job_store_lock:
        if job_id not in _job_store:
            _job_store[job_id] = {
                'job_id': job_id,
                'status': 'queued',
                'progress': 0,
                'message': '',
                'filename': '',
                'duration': 0,
                'frames_analyzed': 0,
                'ffmpeg_available': False,
                'transcript': '',
                'video_heatmap_data': None,
                'text_heatmap_data': None,
                'sentiment': {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0},
                'emotion_summary': {
                    'angry': 0, 'disgust': 0, 'fear': 0,
                    'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0
                },
                'created_at': datetime.now(timezone.utc).isoformat(),
            }
        _job_store[job_id].update(fields)
        _job_store[job_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
    logger.debug(f"Job {job_id} updated — status={fields.get('status')}, progress={fields.get('progress')}")


def get_job(job_id: str) -> dict | None:
    """Retrieve a job record from the in-memory store."""
    with _job_store_lock:
        job = _job_store.get(job_id)
        return dict(job) if job else None  # return a copy to avoid race conditions


def delete_job(job_id: str):
    """Remove a finished job from the store to free memory."""
    with _job_store_lock:
        _job_store.pop(job_id, None)

# ---------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_ffmpeg_path():
    """Detect FFmpeg path — Docker first, then Windows local, then system PATH."""
    is_docker = os.path.exists('/.dockerenv')

    if is_docker:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print(f"✅ Docker: Using system FFmpeg at {ffmpeg_path}")
            return ffmpeg_path
        raise RuntimeError("FFmpeg not found in Docker container! Check Dockerfile installation.")
    else:
        windows_path = r'C:\ffmpeg\ffmpeg-n8.0-latest-win64-gpl-8.0\bin\ffmpeg.exe'
        if os.path.exists(windows_path):
            print("✅ Local: Using Windows FFmpeg")
            return windows_path

        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print("✅ Local: Found FFmpeg in PATH")
            return ffmpeg_path

        raise RuntimeError("FFmpeg not found locally. Install FFmpeg or use Docker.")


# Initialize FFmpeg
FFMPEG_PATH = get_ffmpeg_path()
FFMPEG_AVAILABLE = FFMPEG_PATH is not None
print(f"Using FFmpeg path: {FFMPEG_PATH}")
logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}, path: {FFMPEG_PATH}")


def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to Base64 string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf,
                    format='png',
                    bbox_inches='tight',
                    dpi=dpi,
                    facecolor=fig.get_facecolor(),
                    edgecolor='none',
                    transparent=False)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        logger.debug(f"Converted figure to Base64 ({len(img_str)} characters)")
        return img_str
    except Exception as e:
        logger.error(f"Error converting figure to Base64: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase timeouts (seconds) — user-facing messages emitted when exceeded
# ---------------------------------------------------------------------------
PHASE_TIMEOUTS = {
    'video_load':      15,
    'frame_analysis':  60,   # total budget for ALL frames
    'heatmap_gen':     30,
    'speech_to_text':  40,
    'sentiment':       20,
}

# Hard timeout per individual FER detect_emotions() call.
# MTCNN on CPU can take 5-15s per frame on Render's free tier.
# If a single call exceeds this, we skip the frame and move on.
PER_FRAME_TIMEOUT = 8  # seconds


def analyze_video(video_path: str, job_id: str, filename: str):
    """
    Main analysis pipeline — all state written to the in-memory job store.
    Phases are clearly labelled and each has a timeout with a user-facing message.
    """
    detector = None
    vidcap = None

    try:
        save_job(job_id, filename=filename, status='processing', progress=5,
                 message='🚀 Starting analysis pipeline...')

        # ── Phase 1: Load video ────────────────────────────────────────────
        save_job(job_id, status='processing', progress=8,
                 message='📂 Phase 1/5 — Loading video file...')

        clip = None
        duration = 0
        phase_start = time.time()

        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            logger.info(f"Video duration: {duration}s")

            if time.time() - phase_start > PHASE_TIMEOUTS['video_load']:
                raise TimeoutError("Video loading took too long.")

            if duration > 25:
                raise ValueError(f"Video exceeds 25-second limit ({duration:.1f}s). "
                                 "Please upload a shorter clip.")

            save_job(job_id, duration=duration, status='processing', progress=12,
                     message=f'✅ Phase 1/5 — Video loaded ({duration:.1f}s). '
                              f'Starting facial analysis...')
        except TimeoutError as te:
            raise RuntimeError(
                f"⏱️ Phase 1 timed out: Video file took too long to load "
                f"(>{PHASE_TIMEOUTS['video_load']}s). "
                "Try a smaller file or different format."
            )
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"❌ Phase 1 failed: Could not read video — {str(e)}")
        finally:
            if clip:
                try:
                    clip.reader.close()
                    if clip.audio and clip.audio.reader:
                        clip.audio.reader.close_proc()
                except Exception:
                    pass
                del clip

        # ── Phase 2: Facial emotion detection ─────────────────────────────
        save_job(job_id, status='processing', progress=15,
                 message='😐 Phase 2/5 — Initialising face detector (this may take a moment)...')

        # Use Haar-cascade fallback (no MTCNN) on deployed servers — MTCNN on
        # CPU is extremely slow (15-30s per frame) and causes the phase to time out.
        # MTCNN is only worth using if you have a GPU.
        is_docker = os.path.exists('/.dockerenv')
        use_mtcnn = not is_docker  # MTCNN locally only; Haar on server
        try:
            detector = FER(mtcnn=use_mtcnn)
            logger.info(f"FER initialised — mtcnn={use_mtcnn}")
        except Exception as init_err:
            logger.warning(f"FER init failed ({init_err}), falling back to Haar")
            try:
                detector = FER(mtcnn=False)
            except Exception as fallback_err:
                raise RuntimeError(
                    f'❌ Phase 2 failed: Could not initialise face detector — {fallback_err}'
                )

        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            raise RuntimeError('❌ Phase 2 failed: Could not open video for frame extraction.')

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 60:
            fps = 30  # Sensible default

        frames_to_process = min(int(duration), 25)
        emotions_data = []
        phase_start = time.time()
        timed_out_frames = 0

        logger.info(f"Processing {frames_to_process} frames at {fps:.1f} FPS "
                    f"(per-frame cap: {PER_FRAME_TIMEOUT}s, "
                    f"total cap: {PHASE_TIMEOUTS['frame_analysis']}s)")

        for i in range(frames_to_process):
            # ── Total phase budget check ───────────────────────────────────
            elapsed = time.time() - phase_start
            if elapsed > PHASE_TIMEOUTS['frame_analysis']:
                logger.warning(
                    f"Phase 2 total timeout reached at frame {i}/{frames_to_process} "
                    f"after {elapsed:.1f}s"
                )
                save_job(job_id, status='processing', progress=50,
                         message=(
                             f'⚠️ Phase 2/5 — Face analysis hit the {PHASE_TIMEOUTS["frame_analysis"]}s '
                             f'time limit (processed {i}/{frames_to_process} frames, '
                             f'{timed_out_frames} frames timed out individually). '
                             'Continuing to next phase with partial data...'
                         ))
                break

            try:
                frame_pos = int(i * fps)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = vidcap.read()

                if ret and frame is not None:
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                             f'temp_{job_id}_{i}.jpg')
                    cv2.imwrite(temp_path, frame)

                    try:
                        img = plt.imread(temp_path)

                        # ── Hard per-frame timeout via ThreadPoolExecutor ──
                        # detect_emotions() is a blocking C/PyTorch call.
                        # future.result(timeout=N) lets us abandon it if it
                        # exceeds PER_FRAME_TIMEOUT without hanging the loop.
                        _executor = ThreadPoolExecutor(max_workers=1)
                        _future = _executor.submit(detector.detect_emotions, img)
                        _executor.shutdown(wait=False)  # don't block on cleanup

                        try:
                            detected = _future.result(timeout=PER_FRAME_TIMEOUT)
                            if detected:
                                emotions_data.extend(detected)
                        except FutureTimeoutError:
                            timed_out_frames += 1
                            logger.warning(
                                f"Frame {i} FER call exceeded {PER_FRAME_TIMEOUT}s — skipping"
                            )
                        except Exception as fer_err:
                            logger.debug(f"Frame {i} FER error: {fer_err}")

                    except Exception as read_err:
                        logger.debug(f"Frame {i} image read error: {read_err}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                # Progress update every 5 frames
                if i % 5 == 0:
                    pct = 15 + ((i + 1) / frames_to_process * 35)
                    elapsed_now = time.time() - phase_start
                    save_job(job_id, status='processing',
                             progress=int(min(pct, 50)),
                             message=(
                                 f'🔍 Phase 2/5 — Analysing faces: '
                                 f'frame {i+1}/{frames_to_process} '
                                 f'({elapsed_now:.0f}s elapsed)...'
                             ))

            except Exception as frame_err:
                logger.debug(f"Frame {i} outer error: {frame_err}")
                continue

        if vidcap:
            vidcap.release()

        logger.info(
            f"Phase 2 complete — detections: {len(emotions_data)}, "
            f"timed-out frames: {timed_out_frames}, "
            f"elapsed: {time.time() - phase_start:.1f}s"
        )

        # ── Phase 3: Generate emotion heatmap ─────────────────────────────
        save_job(job_id, status='processing', progress=52,
                 message='📊 Phase 3/5 — Generating emotion heatmap...')

        cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        video_heatmap_data = None
        emotion_summary = {col: 0.0 for col in cols}
        frames_analyzed = 0

        phase_start = time.time()
        try:
            if emotions_data:
                rows = [e['emotions'] for e in emotions_data]
                df_emotions = pd.DataFrame(rows, columns=cols)
                frames_analyzed = len(df_emotions)
                for col in cols:
                    emotion_summary[col] = float(df_emotions[col].mean())
            else:
                logger.warning("No faces detected — using zero heatmap")
                df_emotions = pd.DataFrame(np.zeros((5, len(cols))), columns=cols)

            if time.time() - phase_start > PHASE_TIMEOUTS['heatmap_gen']:
                raise TimeoutError("Heatmap generation timed out.")

            fig1, ax1 = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_emotions, annot=True, fmt='.2f', ax=ax1,
                        cmap='YlOrRd', cbar_kws={'label': 'Emotion Intensity'})
            ax1.set_title('Facial Emotion Analysis Heatmap', fontsize=16, pad=20)
            ax1.set_xlabel('Emotions', fontsize=12)
            ax1.set_ylabel('Video Frames', fontsize=12)

            video_heatmap_data = fig_to_base64(fig1)
            plt.close(fig1)

            if video_heatmap_data:
                logger.info(f"Emotion heatmap generated ({len(video_heatmap_data)} chars)")
                save_job(job_id, status='processing', progress=60,
                         message='✅ Phase 3/5 — Emotion heatmap ready. '
                                  'Extracting audio...')
            else:
                save_job(job_id, status='processing', progress=60,
                         message='⚠️ Phase 3/5 — Heatmap rendering failed. '
                                  'Continuing to audio extraction...')

        except TimeoutError:
            save_job(job_id, status='processing', progress=60,
                     message=(
                         f'⏱️ Phase 3/5 timed out: Heatmap generation exceeded '
                         f'{PHASE_TIMEOUTS["heatmap_gen"]}s. '
                         'Skipping and continuing to audio...'
                     ))
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            save_job(job_id, status='processing', progress=60,
                     message=f'❌ Phase 3/5 failed: {str(e)[:100]}. Continuing...')

        # ── Phase 4: Speech-to-text ────────────────────────────────────────
        text_output = ""
        save_job(job_id, status='processing', progress=63,
                 message='🎤 Phase 4/5 — Extracting audio for speech recognition...')

        if FFMPEG_AVAILABLE and FFMPEG_PATH:
            phase_start = time.time()
            try:
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                          f'{job_id}_audio.wav')
                cmd = (f'"{FFMPEG_PATH}" -i "{video_path}" '
                       f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y')

                logger.info(f"Running FFmpeg: {cmd}")
                process = subprocess.run(
                    cmd, shell=True, capture_output=True,
                    timeout=PHASE_TIMEOUTS['speech_to_text'],
                    text=True, encoding='utf-8', errors='ignore'
                )

                if time.time() - phase_start > PHASE_TIMEOUTS['speech_to_text']:
                    raise TimeoutError("Audio extraction timed out.")

                if process.returncode == 0 and os.path.exists(audio_path):
                    save_job(job_id, status='processing', progress=72,
                             message='🗣️ Phase 4/5 — Recognising speech (Google STT)...')
                    try:
                        r = sr.Recognizer()
                        with sr.AudioFile(audio_path) as source:
                            r.adjust_for_ambient_noise(source, duration=0.5)
                            audio = r.record(source, duration=min(30, duration))
                            try:
                                text_output = r.recognize_google(audio)
                                logger.info(f"Speech recognised: {len(text_output)} chars")
                                save_job(job_id, status='processing', progress=80,
                                         message=f'✅ Phase 4/5 — Speech recognised '
                                                  f'({len(text_output)} characters). '
                                                  'Running sentiment analysis...')
                            except sr.UnknownValueError:
                                text_output = ""
                                save_job(job_id, status='processing', progress=80,
                                         message='⚠️ Phase 4/5 — Speech detected but '
                                                  'could not be understood. '
                                                  'Skipping text sentiment...')
                            except sr.RequestError as e:
                                text_output = ""
                                save_job(job_id, status='processing', progress=80,
                                         message=f'⚠️ Phase 4/5 — Google Speech API '
                                                  f'error: {str(e)[:80]}. '
                                                  'Check your internet connection.')
                    except Exception as ae:
                        logger.debug(f"Audio processing error: {ae}")
                        text_output = ""
                        save_job(job_id, status='processing', progress=80,
                                 message='⚠️ Phase 4/5 — Audio processing failed. '
                                          'Skipping text sentiment...')
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                else:
                    logger.warning(f"FFmpeg failed (rc={process.returncode}): "
                                   f"{process.stderr[:200]}")
                    save_job(job_id, status='processing', progress=80,
                             message='⚠️ Phase 4/5 — Audio extraction failed '
                                      '(FFmpeg error). Skipping text sentiment...')

            except subprocess.TimeoutExpired:
                save_job(job_id, status='processing', progress=80,
                         message=(
                             f'⏱️ Phase 4/5 timed out: Audio extraction exceeded '
                             f'{PHASE_TIMEOUTS["speech_to_text"]}s. '
                             'Skipping text sentiment...'
                         ))
            except TimeoutError:
                save_job(job_id, status='processing', progress=80,
                         message=(
                             f'⏱️ Phase 4/5 timed out: Speech-to-text exceeded '
                             f'{PHASE_TIMEOUTS["speech_to_text"]}s. '
                             'Skipping text sentiment...'
                         ))
            except Exception as e:
                logger.debug(f"Speech processing error: {e}")
                save_job(job_id, status='processing', progress=80,
                         message=f'❌ Phase 4/5 failed: {str(e)[:80]}. '
                                  'Skipping text sentiment...')
        else:
            save_job(job_id, status='processing', progress=80,
                     message='ℹ️ Phase 4/5 — Skipped: FFmpeg is not available on '
                              'this machine. Install FFmpeg to enable speech analysis.')

        # ── Phase 5: Text sentiment analysis ──────────────────────────────
        text_heatmap_data = None
        sentiment = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

        has_valid_text = (
            text_output
            and len(text_output.strip()) > 0
            and 'error' not in text_output.lower()
            and 'disabled' not in text_output.lower()
        )

        if has_valid_text:
            save_job(job_id, status='processing', progress=82,
                     message='💬 Phase 5/5 — Running VADER sentiment analysis...')
            phase_start = time.time()
            try:
                nltk.download('vader_lexicon', quiet=True)
                sia = SentimentIntensityAnalyzer()
                sentiment = sia.polarity_scores(text_output)

                logger.info(f"Sentiment — Neg={sentiment['neg']:.3f}, "
                            f"Neu={sentiment['neu']:.3f}, Pos={sentiment['pos']:.3f}, "
                            f"Compound={sentiment['compound']:.3f}")

                if time.time() - phase_start > PHASE_TIMEOUTS['sentiment']:
                    raise TimeoutError("Sentiment analysis timed out.")

                # Determine label
                compound = sentiment['compound']
                if compound >= 0.05:
                    sentiment_label, sentiment_color = 'POSITIVE', '#10b981'
                elif compound <= -0.05:
                    sentiment_label, sentiment_color = 'NEGATIVE', '#ef4444'
                else:
                    sentiment_label, sentiment_color = 'NEUTRAL', '#f59e0b'

                # Chart 1 — Sentiment scores bar chart
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

                categories = ['Negative', 'Neutral', 'Positive']
                values = [sentiment['neg'], sentiment['neu'], sentiment['pos']]
                colors = ['#ef4444', '#f59e0b', '#10b981']

                bars1 = ax1.bar(categories, values, color=colors,
                                edgecolor='white', linewidth=2)
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Score (0 to 1)', fontsize=12, color='white', fontweight='bold')
                ax1.set_title('Text Sentiment Scores', fontsize=16, color='white',
                              fontweight='bold', pad=20)

                for bar in bars1:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.02,
                             f'{h:.3f}', ha='center', va='bottom',
                             color='white', fontsize=12, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', alpha=0.8))
                    if h > 0.1:
                        ax1.text(bar.get_x() + bar.get_width() / 2., h / 2,
                                 f'{h * 100:.1f}%', ha='center', va='center',
                                 color='white', fontsize=11, fontweight='bold')

                # Chart 2 — Compound score indicator
                bars2 = ax2.barh(['Overall'], [compound], color=sentiment_color, height=0.3)
                ax2.set_xlim(-1, 1)
                ax2.set_xlabel('Sentiment Score (-1 to 1)', fontsize=12, color='white',
                               fontweight='bold')
                ax2.set_title('Overall Sentiment Assessment', fontsize=16, color='white',
                              fontweight='bold', pad=20)

                ax2.axvline(x=0, color='white', linestyle='-', linewidth=2, alpha=0.5)
                ax2.axvline(x=0.5, color='#10b981', linestyle='--', linewidth=1, alpha=0.3)
                ax2.axvline(x=-0.5, color='#ef4444', linestyle='--', linewidth=1, alpha=0.3)

                for bar in bars2:
                    w = bar.get_width()
                    label_x = w / 2 if abs(w) > 0.2 else w + (0.05 if w >= 0 else -0.05)
                    label_color = 'white' if abs(w) > 0.2 else sentiment_color
                    ax2.text(label_x, bar.get_y() + bar.get_height() / 2,
                             f'{w:.3f}', ha='center', va='center',
                             color=label_color, fontsize=14, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='#222' if abs(w) <= 0.2 else sentiment_color,
                                       alpha=0.9))

                ax2.text(0, 1.2, sentiment_label, transform=ax2.transAxes,
                         ha='center', va='center', color=sentiment_color,
                         fontsize=18, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#222',
                                   edgecolor=sentiment_color, linewidth=2))

                # Common styling
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#1a1a1a')
                    for spine in ax.spines.values():
                        spine.set_color('white')
                    ax.tick_params(colors='white', which='both', labelsize=11)
                    ax.grid(True, alpha=0.2, color='white', linestyle='--')

                fig2.patch.set_facecolor('#0a0a0a')
                plt.tight_layout(pad=3.0)
                plt.figtext(0.5, 0.01,
                            'Negative ← -1.0 to -0.05 | Neutral -0.05 to 0.05 | Positive 0.05 to 1.0 →',
                            ha='center', fontsize=10, color='#aaa',
                            bbox=dict(boxstyle='round', facecolor='#222', alpha=0.7))

                text_heatmap_data = fig_to_base64(fig2, dpi=120)
                plt.close(fig2)

                if text_heatmap_data:
                    logger.info("✅ Text sentiment visualisation generated successfully")
                    save_job(job_id, status='processing', progress=95,
                             message='✅ Phase 5/5 — Sentiment analysis complete. '
                                      'Finalising results...')
                else:
                    save_job(job_id, status='processing', progress=95,
                             message='⚠️ Phase 5/5 — Sentiment chart rendering failed.')

            except TimeoutError:
                save_job(job_id, status='processing', progress=95,
                         message=(
                             f'⏱️ Phase 5/5 timed out: Sentiment analysis exceeded '
                             f'{PHASE_TIMEOUTS["sentiment"]}s. '
                             'Results may be partial.'
                         ))
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}\n{traceback.format_exc()}")
                save_job(job_id, status='processing', progress=95,
                         message=f'❌ Phase 5/5 failed: {str(e)[:100]}')
        else:
            save_job(job_id, status='processing', progress=95,
                     message='ℹ️ Phase 5/5 — Skipped: No recognised speech to analyse.')

        # ── Finalise ──────────────────────────────────────────────────────
        save_job(
            job_id,
            status='completed',
            progress=100,
            message='🎉 Analysis complete!',
            duration=duration,
            frames_analyzed=frames_analyzed,
            ffmpeg_available=FFMPEG_AVAILABLE,
            transcript=text_output,
            video_heatmap_data=video_heatmap_data,
            text_heatmap_data=text_heatmap_data,
            sentiment=sentiment,
            emotion_summary=emotion_summary,
        )
        logger.info(f"✅ Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ CRITICAL error in job {job_id}: {e}\n{traceback.format_exc()}")
        error_msg = str(e)
        # Prepend emoji if not already there for UI clarity
        if not error_msg.startswith(('❌', '⏱️', '⚠️')):
            error_msg = f'❌ Analysis failed: {error_msg[:200]}'
        save_job(job_id, status='error', progress=0, message=error_msg)

    finally:
        if vidcap:
            vidcap.release()
        if detector:
            try:
                del detector
            except Exception:
                pass
        # Cleanup uploaded video
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Cleaned up video file: {video_path}")
            except Exception:
                logger.warning(f"Could not clean up video file: {video_path}")
        gc.collect()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def serve_index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start analysis thread."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: MP4, AVI, MOV, MKV'}), 400

        job_id = str(uuid.uuid4())
        filename = file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{filename}')

        file.save(video_path)
        logger.info(f"File uploaded: {filename} → {video_path}")

        # Quick duration validation before spawning thread
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.reader.close()
            if clip.audio and clip.audio.reader:
                clip.audio.reader.close_proc()
            del clip

            if duration > 25:
                os.remove(video_path)
                return jsonify({
                    'error': f'Video exceeds 25-second limit ({duration:.1f}s). '
                             'Please upload a shorter clip.'
                }), 400

        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            logger.error(f"Video validation error: {e}")
            return jsonify({'error': 'Invalid video file. Please try another.'}), 400

        # Register job in store before starting thread (prevents race on SSE)
        save_job(job_id, filename=filename, status='queued', progress=0,
                 message='📥 Video uploaded. Waiting to start...',
                 ffmpeg_available=FFMPEG_AVAILABLE)

        thread = threading.Thread(
            target=analyze_video, args=(video_path, job_id, filename), daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Upload successful. Processing started.',
            'ffmpeg_available': FFMPEG_AVAILABLE
        })

    except Exception as e:
        logger.error(f"Upload endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error during upload'}), 500


@app.route('/api/stream/<job_id>')
def stream_progress(job_id):
    """Stream processing progress via Server-Sent Events (polls in-memory store)."""
    def generate():
        last_progress = -1
        last_status = None
        idle_ticks = 0
        max_idle_ticks = 240  # 120 seconds (0.5s × 240)

        try:
            while idle_ticks < max_idle_ticks:
                job_data = get_job(job_id)

                if job_data:
                    current_progress = job_data.get('progress', 0)
                    current_status = job_data.get('status')

                    if current_progress != last_progress or current_status != last_status:
                        payload = {
                            'progress': current_progress,
                            'message': job_data.get('message', ''),
                            'status': current_status,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        last_progress = current_progress
                        last_status = current_status
                        idle_ticks = 0  # Reset timeout on any real update

                    if current_status in ('completed', 'error'):
                        logger.info(f"SSE stream ending — job {job_id} status: {current_status}")
                        break
                else:
                    idle_ticks += 1
                    if idle_ticks % 20 == 0:  # Every 10 seconds
                        _waiting_msg = json.dumps({'progress': 0, 'status': 'waiting', 'message': 'Waiting for job to start...'})
                        yield f"data: {_waiting_msg}\n\n"

                time.sleep(0.5)

            if idle_ticks >= max_idle_ticks:
                timeout_msg = (
                    '⏱️ The analysis job timed out: no progress was received for '
                    '120 seconds. The server may be under heavy load or a phase '
                    'took too long. Please try again with a shorter video.'
                )
                yield f"data: {json.dumps({'progress': 0, 'message': timeout_msg, 'status': 'timeout'})}\n\n"
                logger.warning(f"SSE stream timed out for job {job_id}")

        except GeneratorExit:
            logger.info(f"SSE connection closed by client for job {job_id}")
        except Exception as e:
            logger.error(f"SSE stream error for job {job_id}: {e}")
            _error_msg = json.dumps({'progress': 0, 'status': 'error', 'message': 'Stream error. Please refresh and try again.'})
            yield f"data: {_error_msg}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/results/<job_id>')
def get_results(job_id):
    """Return completed analysis results from the in-memory store."""
    try:
        job_data = get_job(job_id)

        if not job_data:
            logger.warning(f"Results requested for unknown job: {job_id}")
            return jsonify({'error': 'Job not found. It may have expired — please re-upload your video.'}), 404

        if job_data.get('status') == 'error':
            return jsonify({'error': job_data.get('message', 'Analysis failed.')}), 500

        if job_data.get('status') != 'completed':
            return jsonify({'error': 'Analysis is not yet complete. Please wait.'}), 400

        if not job_data.get('video_heatmap_data'):
            logger.error(f"Emotion heatmap data missing for job {job_id}")
            return jsonify({'error': 'Analysis finished but no heatmap was generated. '
                                     'The video may have had no detectable faces.'}), 500

        logger.info(f"✅ Returning results for job {job_id}")
        return jsonify(job_data)

    except Exception as e:
        logger.error(f"Results endpoint error for job {job_id}: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Server error retrieving results'}), 500


@app.route('/api/heatmap/<job_id>/<heatmap_type>')
def get_heatmap(job_id, heatmap_type):
    """Return a heatmap as a Base64 data URL."""
    try:
        job_data = get_job(job_id)

        if not job_data:
            return jsonify({'error': 'Job not found'}), 404

        key_map = {'video': 'video_heatmap_data', 'text': 'text_heatmap_data'}
        if heatmap_type not in key_map:
            return jsonify({'error': 'Invalid heatmap type. Use "video" or "text".'}), 400

        heatmap_data = job_data.get(key_map[heatmap_type])
        if not heatmap_data:
            return jsonify({'error': f'{heatmap_type.capitalize()} heatmap is not available '
                                     'for this job.'}), 404

        return jsonify({'data_url': f'data:image/png;base64,{heatmap_data}'})

    except Exception as e:
        logger.error(f"Heatmap endpoint error: {e}")
        return jsonify({'error': 'Server error'}), 500


@app.route('/api/health')
def health_check():
    """API health check."""
    try:
        return jsonify({
            'status': 'healthy',
            'ffmpeg_available': FFMPEG_AVAILABLE,
            'ffmpeg_path': FFMPEG_PATH,
            'active_jobs': len(_job_store),
            'storage': 'in-memory',
            'timestamp': time.time(),
            'version': '3.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Background file cleanup
# ---------------------------------------------------------------------------

def cleanup_old_files():
    """Remove stale files from the uploads folder (older than 1 hour)."""
    try:
        cutoff = time.time() - 3600
        cleaned = 0
        for fname in os.listdir(app.config['UPLOAD_FOLDER']):
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                try:
                    os.remove(fpath)
                    cleaned += 1
                except Exception:
                    pass
        if cleaned:
            logger.info(f"Cleaned up {cleaned} old file(s) from uploads/")
    except Exception as e:
        logger.debug(f"Cleanup error: {e}")


def cleanup_worker():
    while True:
        time.sleep(300)
        cleanup_old_files()


if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

    logger.info("=" * 60)
    logger.info("Starting Emotion Analysis API  (v3.0 — no-database build)")
    logger.info(f"FFmpeg available : {FFMPEG_AVAILABLE}")
    logger.info(f"Job storage      : in-memory (no external DB)")
    logger.info("=" * 60)

    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, threaded=True)