from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import uuid
import tempfile
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
import atexit
import io
import base64
from PIL import Image
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
import shutil
import ssl

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

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

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', SUPABASE_KEY)

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing Supabase credentials in .env file!")
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in .env file")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
    
    # Test connection
    response = supabase.table('jobs').select('count', count='exact').limit(1).execute()
    logger.info(f"Supabase connection test successful. Jobs count: {response.count}")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def get_ffmpeg_path():
    """
    Simple detection: Always use system FFmpeg in Docker.
    """
    # Check if we're in Docker (container)
    is_docker = os.path.exists('/.dockerenv')
    
    if is_docker:
        # In Docker: always use system-installed FFmpeg
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print(f"✅ Docker: Using system FFmpeg at {ffmpeg_path}")
            return ffmpeg_path
        else:
            # This should never happen if Dockerfile is correct
            raise RuntimeError("FFmpeg not found in Docker container! Check Dockerfile installation.")
    else:
        # Local development: try Windows path
        windows_path = r'C:\\ffmpeg\\ffmpeg-n8.0-latest-win64-gpl-8.0\\bin\\ffmpeg.exe'
        if os.path.exists(windows_path):
            print(f"✅ Local: Using Windows FFmpeg")
            return windows_path
        
        # Fallback: try system FFmpeg if available
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            print(f"✅ Local: Found FFmpeg in PATH")
            return ffmpeg_path
        
        raise RuntimeError("FFmpeg not found locally. Install FFmpeg or use Docker.")

# Initialize
FFMPEG_PATH = get_ffmpeg_path()
FFMPEG_AVAILABLE = FFMPEG_PATH is not None

print(f"Using FFmpeg path: {FFMPEG_PATH}")
logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}, path: {FFMPEG_PATH}")

def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to Base64 string with high quality."""
    try:
        buf = io.BytesIO()
        
        # Save with high quality settings
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

def save_job_to_supabase(job_id, filename, duration, status, progress, message, **kwargs):
    """Save or update job status in Supabase."""
    if not supabase:
        logger.error("Supabase client not initialized")
        return False
    
    try:
        # Prepare data
        data = {
            'job_id': job_id,
            'filename': filename,
            'duration': duration,
            'status': status,
            'progress': progress,
            'message': message,
            'ffmpeg_available': FFMPEG_AVAILABLE,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Add optional fields if provided
        for key, value in kwargs.items():
            if value is not None:
                data[key] = value
        
        # Update completed_at timestamp if job is completed
        if status == 'completed':
            data['completed_at'] = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"Saving job data to Supabase: {job_id}, status: {status}")
        
        # Check if job exists
        existing = supabase.table('jobs').select('*').eq('job_id', job_id).execute()
        
        if existing.data:
            # Update existing job
            response = supabase.table('jobs').update(data).eq('job_id', job_id).execute()
            logger.debug(f"Updated job {job_id} in Supabase")
        else:
            # Create new job
            response = supabase.table('jobs').insert(data).execute()
            logger.debug(f"Created new job {job_id} in Supabase")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving job to Supabase: {e}")
        logger.error(traceback.format_exc())
        return False

def get_job_from_supabase(job_id):
    """Get job status from Supabase."""
    if not supabase:
        return None
    
    try:
        response = supabase.table('jobs').select('*').eq('job_id', job_id).execute()
        
        if response.data and len(response.data) > 0:
            job = response.data[0]
            logger.debug(f"Retrieved job from Supabase: {job_id}")
            
            # Build response dictionary
            result = {
                'status': job.get('status'),
                'progress': job.get('progress', 0),
                'message': job.get('message', ''),
                'video_heatmap_data': job.get('video_heatmap_data'),
                'text_heatmap_data': job.get('text_heatmap_data'),
                'transcript': job.get('transcript', ''),
                'duration': job.get('duration', 0),
                'frames_analyzed': job.get('frames_analyzed', 0),
                'ffmpeg_available': job.get('ffmpeg_available', False)
            }
            
            # Build sentiment dictionary
            sentiment = {
                'neg': job.get('sentiment_neg', 0),
                'neu': job.get('sentiment_neu', 0),
                'pos': job.get('sentiment_pos', 0),
                'compound': job.get('sentiment_compound', 0)
            }
            result['sentiment'] = sentiment
            
            # Build emotion summary dictionary
            emotion_summary = {
                'angry': job.get('emotion_angry', 0),
                'disgust': job.get('emotion_disgust', 0),
                'fear': job.get('emotion_fear', 0),
                'happy': job.get('emotion_happy', 0),
                'sad': job.get('emotion_sad', 0),
                'surprise': job.get('emotion_surprise', 0),
                'neutral': job.get('emotion_neutral', 0)
            }
            result['emotion_summary'] = emotion_summary
            
            return result
        
        logger.warning(f"No job found in Supabase for ID: {job_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting job from Supabase: {e}")
        logger.error(traceback.format_exc())
        return None

def analyze_video(video_path, job_id, filename):
    """Main analysis function."""
    detector = None
    vidcap = None
    
    try:
        # Save initial job status
        save_job_to_supabase(job_id, filename, 0, 'processing', 0, 'Starting analysis...')
        
        # 1. Load video and check duration
        clip = None
        duration = 0
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            logger.info(f"Video duration: {duration}s")
            
            if duration > 25:
                raise ValueError(f"Video exceeds 25 second limit ({duration:.1f}s)")
            
            save_job_to_supabase(job_id, filename, duration, 'processing', 10, f'Video loaded ({duration:.1f}s)')
        except Exception as e:
            logger.error(f"Video loading error: {e}")
            raise ValueError(f"Invalid video file: {str(e)}")
        finally:
            if clip:
                try:
                    clip.reader.close()
                    if clip.audio and clip.audio.reader:
                        clip.audio.reader.close_proc()
                except:
                    pass
                del clip
        
        # 2. Facial emotion analysis
        try:
            detector = FER(mtcnn=True)
        except:
            detector = FER()  # Fallback
        
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 60:
            fps = 30  # Default fallback
        
        # Process 1 frame per second (max 25 frames for 25s video)
        frames_to_process = min(int(duration), 25)
        emotions_data = []
        
        logger.info(f"Processing {frames_to_process} frames at {fps:.1f} FPS")
        
        for i in range(frames_to_process):
            try:
                # Calculate frame position (1 frame per second)
                frame_pos = int(i * fps)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = vidcap.read()
                
                if ret and frame is not None:
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{job_id}_{i}.jpg')
                    cv2.imwrite(temp_path, frame)
                    
                    try:
                        img = plt.imread(temp_path)
                        detected = detector.detect_emotions(img)
                        if detected:
                            emotions_data.extend(detected)
                    except Exception as e:
                        logger.debug(f"Frame {i} processing error: {e}")
                    
                    # Cleanup immediately
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Update progress every 5 frames
                if i % 5 == 0:
                    progress = 10 + ((i + 1) / frames_to_process * 40)
                    save_job_to_supabase(
                        job_id, filename, duration, 'processing', 
                        min(progress, 50), 
                        f'Analyzing frames: {i+1}/{frames_to_process}'
                    )
                
            except Exception as e:
                logger.debug(f"Error in frame {i}: {e}")
                continue
        
        if vidcap:
            vidcap.release()
        
        logger.info(f"Total faces detected: {len(emotions_data)}")
        
        # 3. Create and save emotion heatmap
        save_job_to_supabase(job_id, filename, duration, 'processing', 60, 'Generating emotion heatmap...')
        
        cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        if emotions_data:
            rows = [e['emotions'] for e in emotions_data]
            df_emotions = pd.DataFrame(rows, columns=cols)
            frames_analyzed = len(df_emotions)
            
            # Calculate emotion summary
            emotion_summary = {}
            for col in cols:
                emotion_summary[col] = float(df_emotions[col].mean())
        else:
            # Create dummy data for heatmap
            logger.warning("No faces detected - creating dummy heatmap")
            df_emotions = pd.DataFrame(np.zeros((5, len(cols))), columns=cols)
            frames_analyzed = 0
            emotion_summary = {col: 0.0 for col in cols}
        
        # Generate emotion heatmap and convert to Base64
        video_heatmap_data = None
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_emotions, annot=True, fmt='.2f', ax=ax1, cmap='YlOrRd', 
                       cbar_kws={'label': 'Emotion Intensity'})
            ax1.set_title('Facial Emotion Analysis Heatmap', fontsize=16, pad=20)
            ax1.set_xlabel('Emotions', fontsize=12)
            ax1.set_ylabel('Video Frames', fontsize=12)
            
            video_heatmap_data = fig_to_base64(fig1)
            plt.close(fig1)
            
            if video_heatmap_data:
                logger.info(f"Emotion heatmap generated successfully ({len(video_heatmap_data)} chars)")
            else:
                logger.error("Failed to generate emotion heatmap data")
            
        except Exception as e:
            logger.error(f"Emotion heatmap generation error: {e}")
            logger.error(traceback.format_exc())
        
        # 4. Speech to text
        text_output = ""
        if FFMPEG_AVAILABLE and FFMPEG_PATH:
            save_job_to_supabase(job_id, filename, duration, 'processing', 75, 'Converting speech to text...')
            
            try:
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_audio.wav')
                
                # Use the correct FFmpeg path with quotes
                cmd = f'"{FFMPEG_PATH}" -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
                
                logger.info(f"Running FFmpeg command: {cmd}")
                process = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    timeout=30,
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                if process.returncode == 0 and os.path.exists(audio_path):
                    try:
                        r = sr.Recognizer()
                        with sr.AudioFile(audio_path) as source:
                            r.adjust_for_ambient_noise(source, duration=0.5)
                            audio = r.record(source, duration=min(30, duration))
                            
                            try:
                                text_output = r.recognize_google(audio)
                                logger.info(f"Speech recognized: {len(text_output)} characters")
                            except sr.UnknownValueError:
                                text_output = "Speech detected but could not be understood"
                            except sr.RequestError as e:
                                text_output = f"Speech API error: {e}"
                    
                    except Exception as e:
                        logger.debug(f"Audio processing error: {e}")
                        text_output = "Audio processing failed"
                    
                    # Cleanup audio
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                else:
                    logger.warning(f"FFmpeg failed. Return code: {process.returncode}")
                    logger.warning(f"FFmpeg stderr: {process.stderr[:200]}")
                    text_output = "Audio extraction failed"
                    
            except subprocess.TimeoutExpired:
                logger.warning("Audio extraction timeout")
                text_output = "Audio extraction timeout"
            except Exception as e:
                logger.debug(f"Speech processing error: {e}")
                text_output = "Speech processing failed"
        else:
            text_output = "Speech analysis disabled (FFmpeg not available)"
            logger.info("Skipping speech analysis - FFmpeg not available")
        
        # 5. Text sentiment analysis - WITH VISIBLE VALUE LABELS
        text_heatmap_data = None
        sentiment = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

        if text_output and len(text_output.strip()) > 0 and "error" not in text_output.lower() and "disabled" not in text_output.lower():
            save_job_to_supabase(job_id, filename, duration, 'processing', 85, 'Analyzing text sentiment...')
            
            try:
                # Ensure NLTK data is available
                nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
                nltk.data.path.append(nltk_data_dir)
                
                try:
                    nltk.download('vader_lexicon', quiet=True, download_dir=nltk_data_dir)
                except:
                    pass  # Already downloaded
                
                sia = SentimentIntensityAnalyzer()
                sentiment = sia.polarity_scores(text_output)
                
                logger.info(f"📊 Overall Text Sentiment: Neg={sentiment['neg']:.3f}, Neu={sentiment['neu']:.3f}, Pos={sentiment['pos']:.3f}, Compound={sentiment['compound']:.3f}")
                
                # Create a clean, simple sentiment visualization that ALWAYS shows values
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                
                # ===== CHART 1: Sentiment Scores Bar Chart (ALWAYS VISIBLE) =====
                categories = ['Negative', 'Neutral', 'Positive']
                values = [sentiment['neg'], sentiment['neu'], sentiment['pos']]
                colors = ['#ef4444', '#f59e0b', '#10b981']
                
                bars1 = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Score (0 to 1)', fontsize=12, color='white', fontweight='bold')
                ax1.set_title('Text Sentiment Scores', fontsize=16, color='white', fontweight='bold', pad=20)
                
                # Add value labels ON TOP of each bar
                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', 
                            ha='center', va='bottom',
                            color='white', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222', alpha=0.8))
                
                # Add percentage labels inside bars (alternative)
                for bar, value in zip(bars1, values):
                    height = bar.get_height()
                    if height > 0.1:  # Only show inside if bar is tall enough
                        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                                f'{value*100:.1f}%', 
                                ha='center', va='center',
                                color='white', fontsize=11, fontweight='bold')
                
                # ===== CHART 2: Overall Sentiment Indicator =====
                # Create a gauge/thermometer style indicator
                compound_score = sentiment['compound']
                
                # Determine color based on compound score
                if compound_score >= 0.05:
                    sentiment_color = '#10b981'  # Green
                    sentiment_label = 'POSITIVE'
                elif compound_score <= -0.05:
                    sentiment_color = '#ef4444'  # Red
                    sentiment_label = 'NEGATIVE'
                else:
                    sentiment_color = '#f59e0b'  # Orange
                    sentiment_label = 'NEUTRAL'
                
                # Create a horizontal bar for compound score
                bars2 = ax2.barh(['Overall'], [compound_score], color=sentiment_color, height=0.3)
                ax2.set_xlim(-1, 1)
                ax2.set_xlabel('Sentiment Score (-1 to 1)', fontsize=12, color='white', fontweight='bold')
                ax2.set_title('Overall Sentiment Assessment', fontsize=16, color='white', fontweight='bold', pad=20)
                
                # Add reference lines
                ax2.axvline(x=0, color='white', linestyle='-', linewidth=2, alpha=0.5)
                ax2.axvline(x=0.5, color='#10b981', linestyle='--', linewidth=1, alpha=0.3)
                ax2.axvline(x=-0.5, color='#ef4444', linestyle='--', linewidth=1, alpha=0.3)
                
                # Add value label INSIDE the bar with high contrast
                for bar in bars2:
                    width = bar.get_width()
                    # Position label inside bar if there's enough space
                    if abs(width) > 0.2:
                        label_x = width/2
                        label_color = 'white'
                    else:
                        label_x = width + (0.05 if width >= 0 else -0.05)
                        label_color = sentiment_color
                    
                    ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}', 
                            ha='center', va='center',
                            color=label_color, fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222' if abs(width) <= 0.2 else sentiment_color, alpha=0.9))
                
                # Add sentiment label above the bar
                ax2.text(0, 1.2, sentiment_label, 
                        transform=ax2.transAxes,
                        ha='center', va='center',
                        color=sentiment_color, fontsize=18, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#222', edgecolor=sentiment_color, linewidth=2))
                
                # ===== COMMON STYLING FOR BOTH CHARTS =====
                for ax in [ax1, ax2]:
                    # Dark background
                    ax.set_facecolor('#1a1a1a')
                    
                    # White spines and ticks
                    ax.spines['bottom'].set_color('white')
                    ax.spines['top'].set_color('white')
                    ax.spines['right'].set_color('white')
                    ax.spines['left'].set_color('white')
                    
                    # White tick labels
                    ax.tick_params(colors='white', which='both', labelsize=11)
                    
                    # Grid for better readability
                    ax.grid(True, alpha=0.2, color='white', linestyle='--')
                
                # Set figure background
                fig2.patch.set_facecolor('#0a0a0a')
                
                # Adjust layout with more padding
                plt.tight_layout(pad=3.0)
                
                # Add a footer with explanation
                plt.figtext(0.5, 0.01, 
                        'Negative ← -1.0 to -0.05 | Neutral -0.05 to 0.05 | Positive 0.05 to 1.0 →',
                        ha='center', fontsize=10, color='#aaa',
                        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.7))
                
                # Convert to Base64
                text_heatmap_data = fig_to_base64(fig2, dpi=120)  # Lower DPI for faster rendering
                plt.close(fig2)
                
                if text_heatmap_data:
                    logger.info(f"✅ Text sentiment visualization generated successfully")
                    logger.info(f"   Negative: {sentiment['neg']:.3f} | Neutral: {sentiment['neu']:.3f} | Positive: {sentiment['pos']:.3f}")
                    logger.info(f"   Compound Score: {sentiment['compound']:.3f} ({sentiment_label})")
                else:
                    logger.error("❌ Failed to generate text sentiment visualization")
                    
            except Exception as e:
                logger.error(f"❌ Sentiment analysis error: {e}")
                logger.error(traceback.format_exc())
        
        # 6. Save final results to Supabase
        logger.info(f"Saving final results for job {job_id} to Supabase...")
        
        success = save_job_to_supabase(
            job_id, filename, duration, 'completed', 100, 'Analysis complete!',
            video_heatmap_data=video_heatmap_data,
            text_heatmap_data=text_heatmap_data,
            transcript=text_output,
            sentiment_neg=sentiment['neg'],
            sentiment_neu=sentiment['neu'],
            sentiment_pos=sentiment['pos'],
            sentiment_compound=sentiment['compound'],
            emotion_angry=emotion_summary['angry'],
            emotion_disgust=emotion_summary['disgust'],
            emotion_fear=emotion_summary['fear'],
            emotion_happy=emotion_summary['happy'],
            emotion_sad=emotion_summary['sad'],
            emotion_surprise=emotion_summary['surprise'],
            emotion_neutral=emotion_summary['neutral'],
            frames_analyzed=frames_analyzed
        )
        
        if success:
            logger.info(f"✅ Analysis completed successfully for job {job_id}")
            logger.info(f"Emotion heatmap data: {'Present' if video_heatmap_data else 'Missing'}")
            logger.info(f"Text heatmap data: {'Present' if text_heatmap_data else 'Missing'}")
        else:
            logger.error(f"❌ Failed to save final results for job {job_id} to Supabase")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL Analysis error for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Save error status to Supabase
        error_message = str(e)[:200]  # Limit error message length
        save_job_to_supabase(
            job_id, filename, 0, 'error', 0, 
            f'Analysis failed: {error_message}'
        )
    finally:
        # Cleanup
        if vidcap:
            vidcap.release()
        
        if detector:
            try:
                del detector
            except:
                pass
        
        # Cleanup video file
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Cleaned up video file: {video_path}")
            except:
                logger.warning(f"Could not cleanup video file: {video_path}")
        
        # Force GC
        gc.collect()

from flask import render_template

@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: MP4, AVI, MOV, MKV'}), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        filename = file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{filename}')
        
        file.save(video_path)
        logger.info(f"File uploaded: {filename}, saved as: {video_path}")
        
        # Quick validation
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.reader.close()
            if clip.audio and clip.audio.reader:
                clip.audio.reader.close_proc()
            del clip
            
            if duration > 25:
                os.remove(video_path)
                return jsonify({'error': f'Video exceeds 25 second limit ({duration:.1f}s)'}), 400
                
        except Exception as e:
            os.remove(video_path)
            logger.error(f"Video validation error: {e}")
            return jsonify({'error': 'Invalid video file. Please try another.'}), 400
        
        # Save initial job status
        save_job_to_supabase(job_id, filename, duration, 'queued', 0, 'Video uploaded successfully')
        
        # Start analysis thread
        thread = threading.Thread(target=analyze_video, args=(video_path, job_id, filename))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Upload successful. Processing started.',
            'ffmpeg_available': FFMPEG_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error during upload'}), 500

@app.route('/api/stream/<job_id>')
def stream_progress(job_id):
    """Stream processing progress using Server-Sent Events."""
    def generate():
        last_progress = -1
        last_status = None
        retry_count = 0
        max_retries = 120  # 60 seconds
        
        try:
            while retry_count < max_retries:
                job_data = get_job_from_supabase(job_id)
                
                if job_data:
                    current_progress = job_data.get('progress', 0)
                    current_status = job_data.get('status')
                    
                    # Only send if progress or status changed
                    if (current_progress != last_progress or 
                        current_status != last_status):
                        
                        response_data = {
                            'progress': current_progress,
                            'message': job_data.get('message', ''),
                            'status': current_status
                        }
                        
                        yield f"data: {json.dumps(response_data)}\n\n"
                        
                        last_progress = current_progress
                        last_status = current_status
                        
                        # If completed or error, break
                        if current_status in ['completed', 'error']:
                            logger.info(f"SSE stream ending for job {job_id} with status: {current_status}")
                            break
                    
                    retry_count = 0  # Reset retry count on successful status
                else:
                    retry_count += 1
                    if retry_count % 10 == 0:  # Every 5 seconds
                        yield f"data: {json.dumps({'progress': 0, 'message': 'Waiting for job to start...', 'status': 'waiting'})}\n\n"
                
                time.sleep(0.5)  # Poll every 0.5 seconds
            
            if retry_count >= max_retries:
                yield f"data: {json.dumps({'progress': 0, 'message': 'Job timeout', 'status': 'timeout'})}\n\n"
                logger.warning(f"SSE timeout for job {job_id}")
                
        except GeneratorExit:
            logger.info(f"SSE connection closed for job {job_id}")
        except Exception as e:
            logger.error(f"SSE stream error for job {job_id}: {e}")
            yield f"data: {json.dumps({'progress': 0, 'message': 'Stream error', 'status': 'error'})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/results/<job_id>')
def get_results(job_id):
    """Get analysis results from Supabase."""
    try:
        job_data = get_job_from_supabase(job_id)
        
        if not job_data:
            logger.warning(f"Results requested for unknown job: {job_id}")
            return jsonify({'error': 'Results not available yet'}), 404
        
        if job_data.get('status') != 'completed':
            return jsonify({'error': 'Analysis not complete yet'}), 400
        
        # Check if we have at least the emotion heatmap
        if not job_data.get('video_heatmap_data'):
            logger.error(f"Emotion heatmap data missing for job {job_id}")
            return jsonify({'error': 'Analysis failed - no heatmap generated'}), 500
        
        logger.info(f"✅ Returning results for job {job_id}")
        logger.info(f"Video heatmap data present: {'Yes' if job_data.get('video_heatmap_data') else 'No'}")
        logger.info(f"Text heatmap data present: {'Yes' if job_data.get('text_heatmap_data') else 'No'}")
        
        return jsonify(job_data)
        
    except Exception as e:
        logger.error(f"Results endpoint error for job {job_id}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Server error retrieving results'}), 500

@app.route('/api/heatmap/<job_id>/<heatmap_type>')
def get_heatmap(job_id, heatmap_type):
    """Get heatmap image as Base64 data URL."""
    try:
        job_data = get_job_from_supabase(job_id)
        
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404
        
        if heatmap_type == 'video':
            heatmap_data = job_data.get('video_heatmap_data')
        elif heatmap_type == 'text':
            heatmap_data = job_data.get('text_heatmap_data')
        else:
            return jsonify({'error': 'Invalid heatmap type'}), 400
        
        if not heatmap_data:
            return jsonify({'error': 'Heatmap not available'}), 404
        
        # Return as data URL
        data_url = f"data:image/png;base64,{heatmap_data}"
        return jsonify({'data_url': data_url})
        
    except Exception as e:
        logger.error(f"Heatmap endpoint error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/api/health')
def health_check():
    """API health check."""
    try:
        supabase_status = "connected" if supabase else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'ffmpeg_available': FFMPEG_AVAILABLE,
            'ffmpeg_path': FFMPEG_PATH,
            'supabase': supabase_status,
            'timestamp': time.time(),
            'version': '2.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

def cleanup_old_files():
    """Cleanup old uploaded files."""
    try:
        cutoff = time.time() - 3600  # 1 hour old
        cleaned = 0
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                try:
                    os.remove(filepath)
                    cleaned += 1
                except:
                    pass
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old files")
    except Exception as e:
        logger.debug(f"Cleanup error: {e}")

# Background cleanup thread
def cleanup_worker():
    while True:
        time.sleep(300)  # Every 5 minutes
        cleanup_old_files()


if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    logger.info("=" * 60)
    logger.info("Starting Emotion Analysis API")
    logger.info(f"Version: 2.0 (Docker)")
    logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}")
    logger.info(f"Supabase connected: {supabase is not None}")
    logger.info("=" * 60)
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    host = '0.0.0.0'  # Important for Docker
    
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, threaded=True)