# app.py (Enhanced Version with Modality-Specific Analysis)
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
import os
import random
import uuid
import time
import cv2
import numpy as np
from PIL import Image
import io
import json
import re
import PyPDF2
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
# Import Google Gemini API
from google import genai
import math
# Create Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'emotion-llama-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'm4a'}
app.config['ALLOWED_PROFILE_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Create upload folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'profiles'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'charts'), exist_ok=True)

# Set up seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize Gemini API
GOOGLE_API_KEY = "AIzaSyDpS4b-552Rxsr0HNkUrkGV9ocNsMTd76M"
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
# Add these imports at the top of server.py
import requests
from urllib.parse import quote
import json
import re

def generate_music_recommendations(analysis_result, profile_text=None):
    """Generate personalized music recommendations based on emotional analysis and profile information"""
    try:
        # Groq API setup with hardcoded key
        GROQ_API_KEY = "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"
        GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        
        # YouTube API setup - replace with your actual API key
        YOUTUBE_API_KEY = "AIzaSyA9VLgI6DUOrwz1xTkUav86OiggS53ot7w"
        YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
        
        # Extract key information from the analysis
        primary_emotion = analysis_result.get('primary_emotion', 'neutral')
        emotions = analysis_result.get('emotions', [])
        emotion_scores = []
        for emotion in emotions:
            if isinstance(emotion, dict) and 'name' in emotion and 'score' in emotion:
                emotion_scores.append(f"{emotion['name']}: {emotion['score']}/10")
        
        # Extract key information from behavioral indicators
        behavioral_indicators = analysis_result.get('behavioral_indicators', [])
        
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a music recommendation expert. Based on the following emotional analysis and individual's age, demographics, language, culture, location, nationality - if available, 
        if the person is older individual, consider vintage music or if the person is younger, consider modern music.
        if the person is from another country or culture, consider music from their region.
        suggest 3 specific songs that would resonate with this person.
        Consider the person's age, demographics, language, culture, location, nationality - if available for the Song's language recommendation. 
        EMOTIONAL ANALYSIS:
        Primary Emotion: {primary_emotion}
        Emotion Scores: {', '.join(emotion_scores)}
        Behavioral Indicators: {', '.join(behavioral_indicators)}
        Analysis Summary: {analysis_result.get('analysis_summary', 'No summary available')}
        
        USER PROFILE:
        {profile_text if profile_text else 'No profile information available.'}
        
        Please provide 3 specific song recommendations with the following details for each:
        1. Song title (be specific and accurate)
        2. Artist name (be specific and accurate)
        3. Brief explanation of why this song matches the emotional state and profile
        
        Format your response as a JSON array with objects having these fields: 
        "title" (string), "artist" (string), "explanation" (string)
        
        Only include songs that are likely to be available on YouTube. Be accurate with titles and artist names.
        """
        print(f"Music Recommendation Prompt: {prompt}")
        # Make request to Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful music recommendation assistant that provides personalized suggestions based on emotional analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            llm_response = response_json['choices'][0]['message']['content']
            
            # Extract JSON from response (in case the LLM adds extra text)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', llm_response, re.DOTALL)
            if json_match:
                recommendations_json = json_match.group(0)
            else:
                # Fallback: try to extract anything that looks like JSON
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    recommendations_json = f"[{json_match.group(0)}]"
                else:
                    recommendations_json = "[]"
            
            try:
                recommendations = json.loads(recommendations_json)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response manually
                recommendations = parse_recommendations_manually(llm_response)
            
            # Use YouTube API to get accurate video details
            enhanced_recommendations = []
            for rec in recommendations:
                try:
                    title = rec.get('title', '')
                    artist = rec.get('artist', '')
                    
                    if not title or not artist:
                        continue
                    
                    # Create search query
                    search_query = f"{title} {artist} official"
                    
                    # Call YouTube API to search for the video
                    search_params = {
                        'key': YOUTUBE_API_KEY,
                        'q': search_query,
                        'part': 'snippet',
                        'maxResults': 1,
                        'type': 'video'
                    }
                    
                    search_response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
                    
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        if 'items' in search_data and len(search_data['items']) > 0:
                            video_item = search_data['items'][0]
                            video_id = video_item['id']['videoId']
                            
                            # Create YouTube URL and get the thumbnail
                            rec['youtube_url'] = f"https://www.youtube.com/watch?v={video_id}"
                            rec['thumbnail_url'] = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                            rec['youtube_id'] = video_id
                            
                            # Get the actual title and artist from YouTube for accuracy
                            rec['youtube_title'] = video_item['snippet']['title']
                        else:
                            # Fallback to search URL if no results
                            search_query = f"{title} {artist}"
                            rec['youtube_url'] = f"https://www.youtube.com/results?search_query={quote(search_query)}"
                            rec['thumbnail_url'] = "/static/images/music_placeholder.jpg"
                    else:
                        # Fallback to search URL if API fails
                        search_query = f"{title} {artist}"
                        rec['youtube_url'] = f"https://www.youtube.com/results?search_query={quote(search_query)}"
                        rec['thumbnail_url'] = "/static/images/music_placeholder.jpg"
                    
                    enhanced_recommendations.append(rec)
                except Exception as e:
                    print(f"Error enhancing recommendation with YouTube API: {str(e)}")
                    # Still add the recommendation even if enhancement fails
                    if 'thumbnail_url' not in rec:
                        rec['thumbnail_url'] = "/static/images/music_placeholder.jpg"
                    if 'youtube_url' not in rec:
                        search_query = f"{rec.get('title', '')} {rec.get('artist', '')}"
                        rec['youtube_url'] = f"https://www.youtube.com/results?search_query={quote(search_query)}"
                    enhanced_recommendations.append(rec)
            
            return enhanced_recommendations
        else:
            print(f"Error from Groq API: {response.text}")
            return generate_fallback_recommendations(primary_emotion)
    
    except Exception as e:
        print(f"Error in music recommendation generation: {str(e)}")
        return generate_fallback_recommendations(analysis_result.get('primary_emotion', 'neutral'))

# Add this right after your app initialization
def create_music_placeholder():
    """Create a music placeholder image if it doesn't exist"""
    placeholder_dir = os.path.join(app.static_folder, 'images')
    os.makedirs(placeholder_dir, exist_ok=True)
    
    placeholder_path = os.path.join(placeholder_dir, 'music_placeholder.jpg')
    if not os.path.exists(placeholder_path):
        # Create a simple music placeholder image
        placeholder_img = np.ones((270, 480, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add music icon and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(placeholder_img, "♫", (220, 140), font, 4, (100, 100, 100), 5)
        cv2.putText(placeholder_img, "Music Recommendation", (120, 200), font, 1, (100, 100, 100), 2)
        
        cv2.imwrite(placeholder_path, placeholder_img)
        print(f"Created music placeholder image at {placeholder_path}")
    
    return placeholder_path

# Call this function during app initialization
create_music_placeholder()

def parse_recommendations_manually(llm_response):
    """Manually parse the LLM response if JSON parsing fails"""
    recommendations = []
    # Look for song sections (numbered items usually)
    song_sections = re.split(r'\d+\.\s+', llm_response)
    
    for section in song_sections:
        if len(section.strip()) > 20:  # Ensure it's substantial enough to be a recommendation
            try:
                # Extract song details using common patterns
                title_match = re.search(r'(?:Title|Song):\s*"?([^"\n]+)"?', section, re.IGNORECASE) or \
                             re.search(r'"([^"]+)"\s+by', section) or \
                             re.search(r'([^\n:]+)\s+by\s+', section)
                
                artist_match = re.search(r'(?:Artist|by):\s*([^\n]+)', section, re.IGNORECASE) or \
                              re.search(r'by\s+([^\n]+)', section)
                
                # Try to find YouTube video ID directly
                video_id_match = re.search(r'(?:ID|video ID|YouTube ID):\s*"?([a-zA-Z0-9_-]{11})"?', section, re.IGNORECASE) or \
                                re.search(r'youtube.com/watch\?v=([a-zA-Z0-9_-]{11})', section) or \
                                re.search(r'youtu.be/([a-zA-Z0-9_-]{11})', section)
                
                # Fallback to full URL if ID not found
                url_match = re.search(r'(?:URL|YouTube|Link):\s*(https://[^\s]+)', section, re.IGNORECASE) or \
                           re.search(r'(https://(?:www\.)?youtu[^\s]+)', section)
                
                explanation_match = re.search(r'(?:Explanation|Why|Reason|Justification):\s*([^#]+)', section, re.IGNORECASE) or \
                                   re.search(r'This song ([^#]+)', section)
                
                rec = {
                    "title": title_match.group(1).strip() if title_match else "Unknown Song",
                    "artist": artist_match.group(1).strip() if artist_match else "Unknown Artist",
                    "explanation": explanation_match.group(1).strip() if explanation_match else "Matches your emotional profile."
                }
                
                # Add youtube_id if found, otherwise use URL
                if video_id_match:
                    rec["youtube_id"] = video_id_match.group(1).strip()
                elif url_match:
                    rec["youtube_url"] = url_match.group(1).strip()
                
                recommendations.append(rec)
            except Exception as e:
                print(f"Error parsing recommendation section: {str(e)}")
    
    # If we couldn't extract any recommendations, create a placeholder
    if not recommendations:
        recommendations.append({
            "title": "Couldn't parse recommendations",
            "artist": "Unknown",
            "explanation": "The AI provided recommendations in an unexpected format."
        })
    
    return recommendations

def generate_fallback_recommendations(primary_emotion):
    """Generate fallback recommendations if the API call fails"""
    fallback_map = {
        "happy": [
            {
                "title": "Happy",
                "artist": "Pharrell Williams",
                "youtube_url": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
                "thumbnail_url": "https://img.youtube.com/vi/ZbZSe6N_BXs/mqdefault.jpg",
                "explanation": "This upbeat song matches your positive emotional state."
            },
            {
                "title": "Good Vibrations",
                "artist": "The Beach Boys",
                "youtube_url": "https://www.youtube.com/watch?v=Eab_beh07HU",
                "thumbnail_url": "https://img.youtube.com/vi/Eab_beh07HU/mqdefault.jpg",
                "explanation": "A classic feel-good song to maintain your happy mood."
            }
        ],
        "sad": [
            {
                "title": "Someone Like You",
                "artist": "Adele",
                "youtube_url": "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
                "thumbnail_url": "https://img.youtube.com/vi/hLQl3WQQoQ0/mqdefault.jpg",
                "explanation": "This emotional ballad resonates with feelings of sadness and reflection."
            },
            {
                "title": "Hurt",
                "artist": "Johnny Cash",
                "youtube_url": "https://www.youtube.com/watch?v=8AHCfZTRGiI",
                "thumbnail_url": "https://img.youtube.com/vi/8AHCfZTRGiI/mqdefault.jpg",
                "explanation": "A powerful song about regret and loss that can help process sad emotions."
            }
        ],
        "angry": [
            {
                "title": "Rage Against the Machine",
                "artist": "Killing In The Name",
                "youtube_url": "https://www.youtube.com/watch?v=bWXazVhlyxQ",
                "thumbnail_url": "https://img.youtube.com/vi/bWXazVhlyxQ/mqdefault.jpg",
                "explanation": "Matches the intensity of anger while providing a healthy outlet."
            },
            {
                "title": "Break Stuff",
                "artist": "Limp Bizkit",
                "youtube_url": "https://www.youtube.com/watch?v=ZpUYjpKg9KY",
                "thumbnail_url": "https://img.youtube.com/vi/ZpUYjpKg9KY/mqdefault.jpg",
                "explanation": "Expresses frustration and anger in a cathartic way."
            }
        ],
        "neutral": [
            {
                "title": "Weightless",
                "artist": "Marconi Union",
                "youtube_url": "https://www.youtube.com/watch?v=UfcAVejslrU",
                "thumbnail_url": "https://img.youtube.com/vi/UfcAVejslrU/mqdefault.jpg",
                "explanation": "A balanced, calming track designed to maintain a neutral state of mind."
            },
            {
                "title": "Intro",
                "artist": "The xx",
                "youtube_url": "https://www.youtube.com/watch?v=_VPKfacgXao",
                "thumbnail_url": "https://img.youtube.com/vi/_VPKfacgXao/mqdefault.jpg",
                "explanation": "A contemplative instrumental track that complements a neutral emotional state."
            }
        ]
    }
    
    # Default to neutral if the emotion isn't in our map
    emotion_key = primary_emotion.lower() if isinstance(primary_emotion, str) else "neutral"
    for key in fallback_map.keys():
        if emotion_key in key or key in emotion_key:
            emotion_key = key
            break
    else:
        emotion_key = "neutral"
    
    return fallback_map.get(emotion_key, fallback_map["neutral"])
def extract_profile_text(profile_path):
    """Extract text from a PDF profile"""
    try:
        text = ""
        if profile_path.endswith('.pdf'):
            with open(profile_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif profile_path.endswith('.txt'):
            with open(profile_path, 'r', encoding='utf-8') as file:
                text = file.read()
        return text
    except Exception as e:
        print(f"Error extracting profile text: {str(e)}")
        return "Error extracting profile text"


def extract_video_frames(video_path, num_frames=8):
    """Extract multiple frames from the video for analysis"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    frames = []
    frame_paths = []
    frame_times = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Extract frames at regular intervals throughout the video
    for i in range(num_frames):
        position = i * (total_frames / num_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            # Calculate timestamp for this frame
            timestamp = position / fps if fps > 0 else 0
            frame_times.append(timestamp)
            
            # Save individual frames for detailed analysis
            frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"frame_{uuid.uuid4()}_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    cap.release()
    
    if not frames:
        raise ValueError("Could not extract frames from video")
    
    return frames, frame_paths, frame_times, duration

def detect_faces_with_mediapipe(frame_path, timestamp, emotion_label):
    """Detect facial landmarks using MediaPipe Face Mesh with improved template compatibility"""
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load the image
        img = cv2.imread(frame_path)
        if img is None:
            return None
        
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Process the image with MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(rgb_image)
            
            # Check if faces were detected
            if not results.multi_face_landmarks:
                return None
            
            # Create a copy of the image to draw on
            annotated_image = img.copy()
            
            # Draw face landmarks for each detected face
            face_landmarks_list = []
            detected_emotions = []
            confidence_scores = []
            
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Calculate face bounding box
                x_min = width
                y_min = height
                x_max = 0
                y_max = 0
                
                # Extract key points
                key_points = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    key_points.append({
                        'name': f'point_{idx}',
                        'x': landmark.x,
                        'y': landmark.y
                    })
                    
                    # Update bounding box
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add face rectangle with padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(width, x_max + padding)
                y_max = min(height, y_max + padding)
                
                # Store bounding box
                face_bbox = {
                    'x': x_min,
                    'y': y_min,
                    'w': x_max - x_min,
                    'h': y_max - y_min
                }
                
                # Draw the face mesh annotations on the image
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                # Draw the face contours
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                # Add rectangle and labels
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add labels with renamed text (no mention of model or "expected")
                cv2.putText(
                    annotated_image,
                    f"Analysis: {emotion_label}",  # Changed from "Expected:"
                    (x_min, y_min - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )
                
                # Create a face data structure
                face_data = {
                    'bbox': face_bbox,
                    'key_points': key_points,
                    'detected_emotion': 'unknown',  # Default value
                    'confidence': 5.0  # Default confidence (middle value on 0-10 scale)
                }
                
                face_landmarks_list.append(face_data)
            
            # Try to get emotion with DeepFace
            emotion_scores = {}
            try:
                from deepface import DeepFace
                face_analysis = DeepFace.analyze(
                    img_path=frame_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                if isinstance(face_analysis, list) and len(face_analysis) > 0:
                    for face_idx, face_data in enumerate(face_analysis):
                        if face_idx < len(face_landmarks_list):
                            detected_emotion = face_data.get('dominant_emotion', 'unknown')
                            emotions_dict = face_data.get('emotion', {})
                            
                            # Add to our face data
                            face_landmarks_list[face_idx]['detected_emotion'] = detected_emotion
                            face_landmarks_list[face_idx]['emotion_scores'] = emotions_dict
                            
                            # Store for later use
                            detected_emotions.append(detected_emotion)
                            
                            # Calculate confidence score for the expected emotion (0-10 scale)
                            # Convert percentage scores to 0-10 scale for consistency
                            confidence_value = float(emotions_dict.get(detected_emotion, 0)) / 10.0
                            # Cap at 10 and ensure it's a float
                            confidence_score = min(10.0, float(confidence_value))
                            face_landmarks_list[face_idx]['confidence'] = confidence_score
                            confidence_scores.append(confidence_score)
                            
                            # Add emotion label to the image
                            cv2.putText(
                                annotated_image,
                                f"Recognition: {detected_emotion}",  # Changed from "Detected:"
                                (face_landmarks_list[face_idx]['bbox']['x'], 
                                 face_landmarks_list[face_idx]['bbox']['y'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2
                            )
                            
                            # Store all emotion scores
                            emotion_scores = emotions_dict
            except Exception as e:
                print(f"DeepFace emotion detection error: {e}")
            
            # Save the annotated image
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'mediapipe')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"mediapipe_analysis_{timestamp:.2f}_{uuid.uuid4()}.jpg")
            cv2.imwrite(output_path, annotated_image)
            
            # Calculate average confidence - ensure it's a float value
            if confidence_scores:
                avg_confidence = float(sum(confidence_scores) / len(confidence_scores))
            else:
                # Default confidence if no scores were calculated
                avg_confidence = 5.0
            
            # Determine the dominant detected emotion (if any)
            dominant_detected = "unknown"
            if detected_emotions:
                # Use most frequent emotion
                from collections import Counter
                emotion_counter = Counter(detected_emotions)
                dominant_detected = emotion_counter.most_common(1)[0][0]
            
            # Prepare the result data with top-level attributes for template compatibility
            result = {
                'timestamp': timestamp,
                'expression_name': emotion_label,
                'detected_emotion': dominant_detected,
                'faces': face_landmarks_list,
                'image_path': output_path,
                'num_faces': len(face_landmarks_list),
                'confidence': float(avg_confidence),  # Ensure it's a float
                'emotion_scores': emotion_scores,
                'key_points': face_landmarks_list[0]['key_points'] if face_landmarks_list else []
            }
            
            return result
            
    except Exception as e:
        print(f"Error in MediaPipe analysis: {str(e)}")
        return None

def analyze_face_with_deepface(frame_path, timestamp, emotion_label):
    """Analyze facial features using DeepFace library"""
    try:
        from deepface import DeepFace
        import cv2
        import numpy as np
        
        # Load the image
        img = cv2.imread(frame_path)
        if img is None:
            return None
        
        # Analyze the face
        face_analysis = DeepFace.analyze(
            img_path=frame_path,
            actions=['emotion', 'age', 'gender', 'race'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # If it's a list (multiple faces), take the first one
        if isinstance(face_analysis, list):
            if len(face_analysis) == 0:
                return None
            face_data = face_analysis[0]
        else:
            face_data = face_analysis
        
        # Extract face region and landmarks
        if 'region' in face_data:
            region = face_data['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Create a copy of the image with facial landmarks
            result_img = img.copy()
            
            # Draw rectangle around face
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw detected emotion label
            cv2.putText(
                result_img, 
                f"Detected: {face_data['dominant_emotion']}", 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            
            # Draw expected emotion label from our analysis
            cv2.putText(
                result_img, 
                f"Expected: {emotion_label}", 
                (x, y-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 0, 0), 
                2
            )
            
            # Save the annotated image
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'deepface')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"deepface_analysis_{timestamp:.2f}_{uuid.uuid4()}.jpg")
            cv2.imwrite(output_path, result_img)
            
            # Prepare the facial landmarks data
            landmarks_data = {
                'timestamp': timestamp,
                'expression_name': emotion_label,
                'confidence': face_data['emotion'][emotion_label] if emotion_label in face_data['emotion'] else 0,
                'detected_emotion': face_data['dominant_emotion'],
                'key_points': [],  # DeepFace doesn't provide detailed landmarks
                'region': {
                    'x': x, 'y': y, 'w': w, 'h': h
                },
                'image_path': output_path,
                'age': face_data.get('age', 0),
                'gender': face_data.get('gender', ''),
                'dominant_race': face_data.get('dominant_race', '')
            }
            
            return landmarks_data
        else:
            return None
            
    except Exception as e:
        print(f"Error in DeepFace analysis: {str(e)}")
        return None
    except Exception as e:
        print(f"Error in DeepFace analysis: {str(e)}")
        return None


def detect_gestures_with_mediapipe(frame_path, timestamp, emotion_label):
    """Detect body gestures and hand movements using MediaPipe Pose and Hands with improved visibility checking"""
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe Solutions
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load the image
        img = cv2.imread(frame_path)
        if img is None:
            return None
        
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Create a copy of the image to draw on
        annotated_image = img.copy()
        
        # Define gesture interpretations based on emotion
        gesture_interpretations = {
            'angry': ['Rigid posture', 'Tense shoulders', 'Forward leaning', 'Direct gaze'],
            'happy': ['Relaxed posture', 'Animated expressions', 'Open body language', 'Engaged gestures'],
            'sad': ['Slumped posture', 'Downward gaze', 'Minimal movement', 'Self-soothing gestures'],
            'fearful': ['Protective posture', 'Tense muscles', 'Withdrawal movements', 'Vigilant scanning'],
            'anxious': ['Fidgeting', 'Tense facial muscles', 'Shallow breathing', 'Restless movements'],
            'surprised': ['Raised eyebrows', 'Widened eyes', 'Sudden movement', 'Backward leaning'],
            'disgusted': ['Nose wrinkling', 'Backward leaning', 'Avoidance gestures', 'Tightened mouth'],
            'neutral': ['Balanced posture', 'Steady gaze', 'Even breathing', 'Measured movements']
        }
        
        # Process the image with MediaPipe Pose
        detected_gestures = []
        visible_parts = {}
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
            pose_results = pose.process(rgb_image)
            
            # Check if pose was detected
            if pose_results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Check which body parts are actually visible
                landmarks = pose_results.pose_landmarks.landmark
                
                # Track visibility of key body parts
                visible_parts = {
                    'left_shoulder': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.7,
                    'right_shoulder': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.7,
                    'left_elbow': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].visibility > 0.7,
                    'right_elbow': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > 0.7,
                    'left_wrist': landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.7,
                    'right_wrist': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.7,
                    'face': (landmarks[mp_pose.PoseLandmark.NOSE].visibility > 0.8 and
                            landmarks[mp_pose.PoseLandmark.LEFT_EYE].visibility > 0.8 and
                            landmarks[mp_pose.PoseLandmark.RIGHT_EYE].visibility > 0.8)
                }
                
                # Only detect arm gestures if arms are actually visible
                arms_visible = (visible_parts['left_shoulder'] and visible_parts['right_shoulder'] and 
                               (visible_parts['left_elbow'] or visible_parts['right_elbow']) and
                               (visible_parts['left_wrist'] or visible_parts['right_wrist']))
                
                # Add analysis of visible features
                if visible_parts['face']:
                    # Detect facial posture (head tilt, gaze direction)
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]
                    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                    
                    # Calculate head tilt
                    eye_angle = np.degrees(np.arctan2(
                        right_eye.y - left_eye.y,
                        right_eye.x - left_eye.x
                    ))
                    
                    if abs(eye_angle) > 10:
                        tilt_direction = "right" if eye_angle > 0 else "left"
                        gesture = f"Head tilted {tilt_direction}"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": min(abs(eye_angle)/15 * 10, 10.0),
                            "meaning": f"May indicate curiosity, confusion, or attentiveness depending on context",
                            "confidence": 0.8
                        })
                    
                    # Detect gaze direction (if looking up/down)
                    if nose.y < (left_eye.y + right_eye.y)/2 - 0.03:
                        gesture = "Looking up"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": 7.0,
                            "meaning": "Often indicates recall, contemplation, or consideration of ideas",
                            "confidence": 0.7
                        })
                    elif nose.y > (left_eye.y + right_eye.y)/2 + 0.03:
                        gesture = "Looking down"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": 7.0,
                            "meaning": "May indicate shyness, submission, or focusing on details",
                            "confidence": 0.7
                        })
                
                # Only detect arm gestures if actually visible with high confidence
                if arms_visible:
                    # Check for arms crossed - need to clearly see both arms crossing
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    # Verify actual crossing pattern with high visibility
                    if (left_wrist.visibility > 0.8 and right_wrist.visibility > 0.8 and
                        left_wrist.x > right_shoulder.x and right_wrist.x < left_shoulder.x):
                        gesture = "Arms crossed"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": 8.0,
                            "meaning": "Defensive or closed-off posture; may indicate disagreement or discomfort",
                            "confidence": 0.9
                        })
                    
                    # Check for open arms - requires seeing both arms extended
                    arm_width = abs(left_wrist.x - right_wrist.x)
                    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
                    if (left_wrist.visibility > 0.8 and right_wrist.visibility > 0.8 and 
                        arm_width > shoulder_width * 1.5):
                        gesture = "Open arms"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": 8.0,
                            "meaning": "Welcoming, inclusive, or expressive; often indicates openness or joy",
                            "confidence": 0.85
                        })
        
        # Process the image with MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
            
            hands_results = hands.process(rgb_image)
            
            # Check if hands were detected
            if hands_results.multi_hand_landmarks:
                # Draw hand landmarks for each detected hand
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Get hand label
                    hand_label = "Right" if hands_results.multi_handedness[hand_idx].classification[0].label == "Right" else "Left"
                    
                    # Analyze hand gestures
                    # Check for specific hand gestures with high visibility
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Calculate distance between thumb and index finger
                    thumb_index_distance = np.sqrt(
                        (thumb_tip.x - index_tip.x)**2 + 
                        (thumb_tip.y - index_tip.y)**2
                    )
                    
                    # Detect pointing gesture - extended index finger
                    if thumb_index_distance > 0.1:
                        gesture = f"{hand_label} hand pointing"
                        detected_gestures.append({
                            "gesture_type": gesture,
                            "intensity": 7.0,
                            "meaning": "Directing attention, emphasis, or accusation",
                            "confidence": 0.8
                        })
        
        # If no specific gestures detected, add posture and emotional cues based on emotion
        if not detected_gestures and visible_parts.get('face', False):
            # Choose posture cues based on the emotion
            if emotion_label.lower() in gesture_interpretations:
                possible_gestures = gesture_interpretations[emotion_label.lower()]
                gesture = possible_gestures[len(frame_path) % len(possible_gestures)]  # Pseudo-random selection
                
                detected_gestures.append({
                    "gesture_type": gesture,
                    "intensity": 6.0,
                    "meaning": f"Common body language associated with {emotion_label}",
                    "confidence": 0.6
                })
            else:
                # Generic posture analysis if emotion doesn't match known categories
                detected_gestures.append({
                    "gesture_type": "Facial expression",
                    "intensity": 5.0,
                    "meaning": "Facial expressions often convey emotional states more reliably than other body language",
                    "confidence": 0.5
                })
        
        # Add visibility indicators to the image - helps debug issues
        cv2.putText(
            annotated_image,
            f"Emotion: {emotion_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )
        
        # Add visibility status text
        vis_text = "Parts visible: "
        vis_text += "Face " if visible_parts.get('face', False) else ""
        vis_text += "Arms " if arms_visible else ""
        cv2.putText(
            annotated_image,
            vis_text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1
        )
        
        # Draw rectangles and labels for each detected gesture
        for i, gesture in enumerate(detected_gestures):
            cv2.putText(
                annotated_image,
                f"Gesture: {gesture['gesture_type']}",
                (10, 70 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # Save the annotated image
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'gestures')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"gesture_analysis_{timestamp:.2f}_{uuid.uuid4()}.jpg")
        cv2.imwrite(output_path, annotated_image)
        
        # Return the analysis result
        result = {
            'timestamp': timestamp,
            'expression_name': emotion_label,
            'gestures': detected_gestures,
            'image_path': output_path,
            'num_gestures': len(detected_gestures),
            'confidence': float(np.mean([g['confidence'] for g in detected_gestures])) if detected_gestures else 5.0,
            'visibility': visible_parts
        }
        
        return result
        
    except Exception as e:
        print(f"Error in gesture analysis: {str(e)}")
        return None
    
def generate_gesture_analysis_chart(gesture_data):
    """Generate a visualization of body language and gesture analysis"""
    if not gesture_data or len(gesture_data) < 1:
        return None
    
    import math
    import cv2
    import numpy as np
    
    # Filter data to only include entries with gestures and remove duplicates
    unique_frames = {}
    
    for data in gesture_data:
        # Skip entries without an image path
        if 'image_path' not in data or not os.path.exists(data['image_path']):
            continue
            
        # Skip entries without gestures
        if 'gestures' not in data or not data['gestures']:
            if 'num_gestures' not in data or data['num_gestures'] == 0:
                continue
        
        # Use the timestamp as a key to avoid duplicates
        # Round to nearest second to further reduce similar frames
        key = round(data['timestamp'])
        
        # Only keep entries with the highest number of gestures for each timestamp
        if key not in unique_frames:
            unique_frames[key] = data
        elif data.get('num_gestures', 0) > unique_frames[key].get('num_gestures', 0):
            unique_frames[key] = data
    
    # Get list of unique frame data, sorted by timestamp
    unique_gesture_data = sorted(unique_frames.values(), key=lambda x: x['timestamp'])
    
    # If we don't have any valid frames with gestures, return None
    if not unique_gesture_data:
        return None
    
    # Calculate grid dimensions - make sure we have at most 6 images
    num_images = min(6, len(unique_gesture_data))
    grid_size = math.ceil(math.sqrt(num_images))
    rows = grid_size
    cols = grid_size
    
    # Create a figure for the visualization
    fig = plt.figure(figsize=(15, 10), facecolor='#FAFAFA')
    
    # For each selected timestamp, display the annotated image
    for i, data in enumerate(unique_gesture_data[:num_images]):
        timestamp = data['timestamp']
        expression = data['expression_name']
        
        # Create subplot
        ax = fig.add_subplot(rows, cols, i+1)
        
        # Read the annotated image
        img = cv2.imread(data['image_path'])
        if img is not None:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
            # Add gesture info as text overlay
            if 'gestures' in data and data['gestures']:
                gesture_text = data['gestures'][0]['gesture_type']
                meaning_text = data['gestures'][0]['meaning']
                
                # Add text box for gesture meaning
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                ax.text(0.5, 0.95, gesture_text, 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', horizontalalignment='center',
                       bbox=props)
        
        # Format timestamp
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f'{minutes}:{seconds:02d}'
        
        # Add timestamp and emotion as title
        ax.set_title(f'Time: {time_str} - {expression}', fontsize=10, pad=10)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a main title
    plt.suptitle('Body Language and Gesture Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"gesture_analysis_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename

def get_video_duration(video_path):
    """Get the duration of a video file in seconds"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        return total_frames / fps
    return 0

def get_audio_duration(audio_path):
    """Estimate audio duration using a command line tool or library"""
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 
             'default=noprint_wrappers=1:nokey=1', audio_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT
        )
        return float(result.stdout)
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 60  # Default to 60 seconds if we can't determine duration

def generate_emotion_intensity_chart(emotions_data):
    """Generate a visually appealing bar chart showing emotion intensities"""
    # Sort emotions by score for better visualization
    emotions_data = sorted(emotions_data, key=lambda x: x['score'], reverse=True)
    emotions = [e['name'].capitalize() for e in emotions_data]
    scores = [e['score'] for e in emotions_data]
    
    # Define elegant color palette matching UI
    emotion_colors = {
        'Happy': '#FFD166',      # Vibrant yellow
        'Sad': '#118AB2',        # Soft blue
        'Angry': '#EF476F',      # Soft red
        'Fearful': '#9B5DE5',    # Purple
        'Disgusted': '#06D6A0',  # Teal
        'Surprised': '#F78C6B',  # Orange
        'Contempt': '#073B4C',   # Dark blue
        'Neutral': '#8D99AE',    # Gray blue
        'Anxious': '#E07A5F'     # Terracotta
    }
    
    # Create figure with custom styling
    plt.figure(figsize=(10, 6), facecolor='#FAFAFA')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FAFAFA')
    
    # Set colors based on emotion names with gradient effect
    colors = [emotion_colors.get(e, '#2c3e50') for e in emotions]
    
    # Create horizontal bars with gradient effect
    for i, (emotion, score, color) in enumerate(zip(emotions, scores, colors)):
        # Create gradient effect
        gradient = np.linspace(0.7, 1, 100)
        gradient_colors = [(np.array(matplotlib.colors.to_rgb(color)) * g) for g in gradient]
        
        # Plot bar with gradient
        for j, gc in enumerate(gradient_colors):
            width = score * (j+1)/len(gradient_colors) * 0.1
            ax.barh(emotion, width, left=j/10, color=gc, height=0.5, alpha=0.8)
            
        # Add score label
        ax.text(score + 0.2, i, f'{score}/10', va='center', fontsize=11, 
                fontweight='bold', color='#343a40')
    
    # Style improvements
    ax.set_xlim(0, 11)
    ax.set_xlabel('Score (0-10)', fontsize=12, fontweight='bold', color='#495057')
    ax.set_title('Emotion Intensity', fontsize=16, fontweight='bold', color='#343a40', pad=20)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    
    # Grid styling
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#cccccc')
    ax.tick_params(axis='both', colors='#555555')
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    
    # Save with high DPI for crispness
    chart_filename = f"intensity_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename

def generate_emotion_timeline_chart(timestamps, emotions, duration):
    """Generate an elegant timeline chart showing emotions over time"""
    # Define sophisticated color palette
    emotion_colors = {
        'happy': '#FFD166',
        'sad': '#118AB2',
        'angry': '#EF476F',
        'fearful': '#9B5DE5',
        'disgusted': '#06D6A0',
        'surprised': '#F78C6B',
        'contempt': '#073B4C',
        'neutral': '#8D99AE',
        'anxious': '#E07A5F'
    }
    
    # Create figure with styling
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FAFAFA')
    
    # Plot emotions with enhanced visuals
    emotion_y_positions = {}
    used_emotions = set()
    
    # First determine y-positions to avoid overlaps
    for emotion in emotions:
        emotion_lower = emotion.lower()
        if emotion_lower not in used_emotions:
            if not emotion_y_positions:
                emotion_y_positions[emotion_lower] = 1
            else:
                # Assign positions with spacing
                taken_positions = list(emotion_y_positions.values())
                new_position = 1
                while new_position in taken_positions:
                    new_position += 0.5
                emotion_y_positions[emotion_lower] = new_position
            used_emotions.add(emotion_lower)
    
    # Create line connecting points of same emotion
    for emotion_name in emotion_y_positions.keys():
        emotion_indices = [i for i, e in enumerate(emotions) if e.lower() == emotion_name]
        if len(emotion_indices) > 1:
            emotion_times = [timestamps[i] for i in emotion_indices]
            emotion_y = [emotion_y_positions[emotion_name]] * len(emotion_indices)
            ax.plot(emotion_times, emotion_y, '-', color=emotion_colors.get(emotion_name, '#2c3e50'), 
                   alpha=0.5, linewidth=2)
    
    # Plot points with hover effects
    for ts, emotion in zip(timestamps, emotions):
        emotion_lower = emotion.lower()
        y_position = emotion_y_positions[emotion_lower]
        color = emotion_colors.get(emotion_lower, '#2c3e50')
        
        # Create glow effect with multiple circles
        for size, alpha in [(24, 0.2), (18, 0.3), (12, 0.5)]:
            ax.scatter(ts, y_position, s=size, color=color, alpha=alpha, zorder=4)
        
        # Main point
        ax.scatter(ts, y_position, s=80, color=color, edgecolor='white', 
                  linewidth=2, alpha=0.9, zorder=5)
        
        # Add emotion label with rotation for less overlap
        if emotion != "neutral":  # Don't label every neutral point to reduce clutter
            ax.annotate(emotion.capitalize(), (ts, y_position + 0.15), rotation=0, 
                       ha='center', fontsize=9, fontweight='bold', color='#343a40')
    
    # Format time axis with proper time formatting
    def format_time(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes}:{seconds:02d}'
    
    formatter = matplotlib.ticker.FuncFormatter(format_time)
    ax.xaxis.set_major_formatter(formatter)
    
    # Configure chart appearance
    ax.set_ylim(0, max(emotion_y_positions.values()) + 1)
    ax.set_xlim(0, duration)
    ax.set_xlabel('Timeline (minutes:seconds)', fontsize=12, fontweight='bold', color='#495057')
    ax.set_title('Emotional Journey Timeline', fontsize=16, fontweight='bold', color='#343a40', pad=20)
    
    # Remove y-axis ticks and labels as they're not meaningful
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Custom stylish legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=emotion.capitalize()) 
                      for emotion, color in emotion_colors.items() 
                      if emotion in [e.lower() for e in emotions]]
    
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(legend_elements)),
             frameon=True, fancybox=True, shadow=True)
    
    # Remove spines and add subtle grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#dddddd')
    
    # Grid styling
    ax.grid(axis='x', linestyle='--', alpha=0.3, color='#cccccc')
    ax.tick_params(axis='x', colors='#555555')
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    
    # Save with high DPI for crispness
    chart_filename = f"timeline_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename

def generate_facial_expression_radar(modality_data):
    """Generate a radar chart for facial expressions"""
    # Extract visual cues focused on facial expressions
    facial_cues = []
    for item in modality_data:
        if any(term in item['cue'].lower() for term in 
              ['face', 'eye', 'mouth', 'brow', 'lip', 'expression', 'smile', 'frown']):
            facial_cues.append(item)
    
    if len(facial_cues) < 3:  # Need at least 3 points for a meaningful radar chart
        return None
    
    # Define categories and scores
    categories = [cue['cue'][:25] + '...' if len(cue['cue']) > 25 else cue['cue'] 
                 for cue in facial_cues[:8]]  # Limit to 8 for readability
    
    # Generate intensity scores based on text analysis
    # This is a simplified approach - in a real app, you'd have actual intensity values
    intensity_terms = {
        'high': 8, 'strong': 8, 'intense': 9, 'very': 7, 'clearly': 7,
        'moderate': 5, 'medium': 5, 'somewhat': 4,
        'slight': 3, 'subtle': 2, 'minimal': 1, 'barely': 1
    }
    
    scores = []
    for cue in facial_cues[:8]:
        score = 5  # Default moderate score
        text = cue['interpretation'].lower()
        for term, value in intensity_terms.items():
            if term in text:
                score = value
                break
        scores.append(score)
    
    # Create radar chart
    fig = plt.figure(figsize=(8, 8), facecolor='#FAFAFA')
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    
    # Close the circle
    categories.append(categories[0])
    scores.append(scores[0])
    angles.append(angles[0])
    
    # Plot data
    ax.plot(angles, scores, 'o-', linewidth=2, color='#3498db')
    
    # Fill area
    ax.fill(angles, scores, alpha=0.25, color='#3498db')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=9)
    
    # Remove radial labels and set range
    ax.set_yticklabels([])
    ax.set_ylim(0, 10)
    
    # Add subtle grid
    ax.grid(color='#dddddd', alpha=0.3)
    
    # Title
    plt.title('Facial Expression Analysis', fontsize=15, fontweight='bold', color='#343a40', pad=20)
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"facial_radar_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename

def generate_vocal_pattern_chart(audio_data, timeline_data):
    """Generate a visualization of vocal patterns over time"""
    if not audio_data or len(audio_data) < 2:
        return None
    
    # Extract time points from timeline
    if timeline_data:
        time_points = [entry['timestamp'] for entry in timeline_data]
        emotions = [entry['emotion'] for entry in timeline_data]
    else:
        # Create artificial timeline if not available
        time_points = np.linspace(0, 100, len(audio_data))
        emotions = ['neutral'] * len(audio_data)
    
    # Define vocal properties to track
    properties = ['pitch', 'volume', 'rate', 'tone']
    
    # Generate data based on audio cue text analysis
    # In a real app, you'd extract actual audio metrics
    data = {prop: [] for prop in properties}
    time_markers = []
    
    for i, cue in enumerate(audio_data):
        time_position = time_points[min(i, len(time_points)-1)]
        time_markers.append(time_position)
        
        cue_text = cue['cue'].lower() + ' ' + cue['interpretation'].lower()
        
        # Analyze text for pitch indicators
        if 'high pitch' in cue_text or 'higher pitch' in cue_text:
            data['pitch'].append(7)
        elif 'low pitch' in cue_text or 'lower pitch' in cue_text:
            data['pitch'].append(3)
        else:
            data['pitch'].append(5)
        
        # Analyze for volume
        if any(term in cue_text for term in ['loud', 'shouting', 'yelling']):
            data['volume'].append(8)
        elif any(term in cue_text for term in ['quiet', 'soft', 'whisper']):
            data['volume'].append(2)
        else:
            data['volume'].append(5)
        
        # Analyze for speaking rate
        if any(term in cue_text for term in ['fast', 'rapid', 'quick']):
            data['rate'].append(8)
        elif any(term in cue_text for term in ['slow', 'deliberate', 'measured']):
            data['rate'].append(3)
        else:
            data['rate'].append(5)
        
        # Analyze for tone
        if any(term in cue_text for term in ['warm', 'friendly', 'kind']):
            data['tone'].append(8)
        elif any(term in cue_text for term in ['harsh', 'cold', 'stern']):
            data['tone'].append(2)
        else:
            data['tone'].append(5)
    
    # Create multi-line chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FAFAFA')
    
    # Color palette
    colors = {
        'pitch': '#3498db',
        'volume': '#e74c3c',
        'rate': '#2ecc71',
        'tone': '#9b59b6'
    }
    
    # Plot each property
    for prop in properties:
        ax.plot(time_markers, data[prop], 'o-', color=colors[prop], linewidth=2, 
               label=prop.capitalize(), alpha=0.8)
    
    # Mark emotions at the bottom
    emotion_colors = {
        'happy': '#FFD166',
        'sad': '#118AB2',
        'angry': '#EF476F',
        'fearful': '#9B5DE5',
        'surprised': '#F78C6B',
        'neutral': '#8D99AE'
    }
    
    # Add emotion markers
    for time, emotion in zip(time_points, emotions):
        color = emotion_colors.get(emotion.lower(), '#8D99AE')
        ax.scatter(time, 0.5, color=color, s=100, zorder=5, 
                  marker='o', edgecolor='white', linewidth=1.5)
    
    # Style the chart
    ax.set_xlim(min(time_markers)-5, max(time_markers)+5)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold', color='#495057')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold', color='#495057')
    ax.set_title('Vocal Pattern Analysis', fontsize=15, fontweight='bold', color='#343a40', pad=20)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    
    # Grid styling
    ax.grid(linestyle='--', alpha=0.3, color='#cccccc')
    
    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, 
             frameon=True, fancybox=True, shadow=True)
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"vocal_pattern_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename



def generate_facial_landmarks_chart(landmarks_data):
    """Generate a visualization of facial landmarks showing only unique frames with detected faces"""
    if not landmarks_data or len(landmarks_data) < 1:
        return None
    
    import math
    import cv2
    import numpy as np
    
    # Filter data to only include entries with faces and remove duplicates
    unique_frames = {}
    
    for data in landmarks_data:
        # Skip entries without an image path
        if 'image_path' not in data or not os.path.exists(data['image_path']):
            continue
            
        # Skip entries without faces
        if 'faces' not in data or not data['faces']:
            if 'num_faces' not in data or data['num_faces'] == 0:
                continue
        
        # Use the timestamp as a key to avoid duplicates
        # Round to nearest second to further reduce similar frames
        key = round(data['timestamp'])
        
        # Only keep entries with the highest number of faces for each timestamp
        if key not in unique_frames:
            unique_frames[key] = data
        elif data.get('num_faces', 0) > unique_frames[key].get('num_faces', 0):
            unique_frames[key] = data
    
    # Get list of unique frame data, sorted by timestamp
    unique_landmarks_data = sorted(unique_frames.values(), key=lambda x: x['timestamp'])
    
    # If we don't have any valid frames with faces, return None
    if not unique_landmarks_data:
        return None
    
    # Calculate grid dimensions - make sure we have at most 6 images
    num_faces = min(6, len(unique_landmarks_data))
    grid_size = math.ceil(math.sqrt(num_faces))
    rows = grid_size
    cols = grid_size
    
    # Create a figure for the visualization
    fig = plt.figure(figsize=(15, 10), facecolor='#FAFAFA')
    
    # For each selected timestamp, display the annotated image
    for i, data in enumerate(unique_landmarks_data[:num_faces]):
        timestamp = data['timestamp']
        expression = data['expression_name']
        
        # Create subplot
        ax = fig.add_subplot(rows, cols, i+1)
        
        # Read the annotated image
        img = cv2.imread(data['image_path'])
        if img is not None:
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        
        # Format timestamp
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f'{minutes}:{seconds:02d}'
        
        # Add timestamp and emotion as title
        ax.set_title(f'Time: {time_str} - {expression}', fontsize=10, pad=10)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a main title
    plt.suptitle('Facial Landmarks Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"facial_landmarks_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename

def generate_verbal_sentiment_flow(verbal_data, timeline_data=None):
    """Generate a sentiment flow visualization for verbal content"""
    if not verbal_data or len(verbal_data) < 2:
        return None
    
    # Extract content and sentiment
    contents = [item['content'] for item in verbal_data]
    
    # Generate sentiment scores (-1 to 1 scale)
    # In a real app, you'd use a proper sentiment analysis model
    sentiment_words = {
        'positive': ['happy', 'love', 'joy', 'excited', 'good', 'great', 'wonderful', 'pleased'],
        'negative': ['sad', 'anger', 'angry', 'upset', 'bad', 'terrible', 'annoyed', 'afraid']
    }
    
    sentiments = []
    for item in verbal_data:
        text = item['content'].lower() + ' ' + item['emotional_meaning'].lower()
        pos_count = sum(word in text for word in sentiment_words['positive'])
        neg_count = sum(word in text for word in sentiment_words['negative'])
        
        if pos_count > neg_count:
            score = min(0.5 + (pos_count - neg_count) * 0.1, 1.0)
        elif neg_count > pos_count:
            score = max(-0.5 - (neg_count - pos_count) * 0.1, -1.0)
        else:
            score = 0
            
        sentiments.append(score)
    
    # Create a smooth flow visualization
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FAFAFA')
    
    # X-axis represents sequence
    x = np.arange(len(sentiments))
    
    # Plot sentiment flow
    ax.plot(x, sentiments, 'o-', linewidth=3, color='#3498db', alpha=0.8)
    
    # Fill areas above/below zero with different colors
    ax.fill_between(x, sentiments, 0, where=(np.array(sentiments) > 0), 
                   color='#2ecc71', alpha=0.3)
    ax.fill_between(x, sentiments, 0, where=(np.array(sentiments) < 0), 
                   color='#e74c3c', alpha=0.3)
    
    # Add content labels (shortened)
    for i, (content, sentiment) in enumerate(zip(contents, sentiments)):
        # Shorten content for label
        short_content = content[:20] + '...' if len(content) > 20 else content
        
        # Alternate label positions to avoid overlap
        vert_pos = 0.1 if i % 2 == 0 else -0.1
        vert_pos = sentiment + (vert_pos if sentiment > 0 else -vert_pos)
        
        ax.annotate(short_content, (i, vert_pos), 
                   xytext=(0, 10 if sentiment > 0 else -10),
                   textcoords='offset points',
                   ha='center', va='center' if sentiment > 0 else 'top',
                   fontsize=8, color='#343a40',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='#dddddd'))
    
    # Style the chart
    ax.set_xlim(-0.5, len(sentiments) - 0.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Message Sequence', fontsize=12, fontweight='bold', color='#495057')
    ax.set_ylabel('Sentiment (Negative to Positive)', fontsize=12, fontweight='bold', color='#495057')
    ax.set_title('Verbal Content Sentiment Flow', fontsize=15, fontweight='bold', color='#343a40', pad=20)
    
    # Hide x-ticks but keep zero line for reference
    ax.set_xticks([])
    ax.axhline(y=0, color='#cccccc', linestyle='-', linewidth=1)
    
    # Add sentiment guide
    ax.text(len(sentiments)/2, 1.1, 'Positive Sentiment', ha='center', color='#2ecc71', fontweight='bold')
    ax.text(len(sentiments)/2, -1.1, 'Negative Sentiment', ha='center', color='#e74c3c', fontweight='bold')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"sentiment_flow_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename


def generate_raw_acoustic_features_chart(raw_audio_features):
    """Generate a visualization of direct acoustic measurements over time"""
    if not raw_audio_features or len(raw_audio_features) < 2:
        return None
    
    # Extract data
    timestamps = [item['timestamp'] for item in raw_audio_features]
    pitches = [item['pitch'] for item in raw_audio_features]
    volumes = [item['volume'] for item in raw_audio_features]
    speech_rates = [item['speech_rate'] for item in raw_audio_features]
    vocal_tensions = [item.get('vocal_tension', 5) for item in raw_audio_features]  # Default to 5 if not available
    
    # Create multi-line chart with advanced styling
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#FAFAFA')
    
    # Define sophisticated color palette
    colors = {
        'pitch': '#3b82f6',       # Blue
        'volume': '#ef4444',      # Red
        'speech_rate': '#10b981', # Green
        'vocal_tension': '#8b5cf6' # Purple
    }
    
    # Plot each acoustic feature with custom styling
    ax.plot(timestamps, pitches, 'o-', color=colors['pitch'], linewidth=2.5, 
           label='Pitch', alpha=0.85, markersize=6)
    ax.plot(timestamps, volumes, 'o-', color=colors['volume'], linewidth=2.5, 
           label='Volume', alpha=0.85, markersize=6)
    ax.plot(timestamps, speech_rates, 'o-', color=colors['speech_rate'], linewidth=2.5, 
           label='Speech Rate', alpha=0.85, markersize=6)
    ax.plot(timestamps, vocal_tensions, 'o-', color=colors['vocal_tension'], linewidth=2.5, 
           label='Vocal Tension', alpha=0.85, markersize=6)
    
    # Add range indicators
    for i, ts in enumerate(timestamps):
        for feature, values, color in [
            ('pitch', pitches, colors['pitch']),
            ('volume', volumes, colors['volume']),
            ('speech_rate', speech_rates, colors['speech_rate']),
            ('vocal_tension', vocal_tensions, colors['vocal_tension'])
        ]:
            if i > 0:
                # Add semi-transparent area to show transitions
                ax.fill_between(
                    [timestamps[i-1], ts], 
                    [values[i-1], values[i]], 
                    alpha=0.1, 
                    color=color
                )
    
    # Style the chart
    ax.set_xlim(min(timestamps)-5, max(timestamps)+5)
    ax.set_ylim(0, 11)  # 0-10 scale with some padding
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold', color='#334155')
    ax.set_ylabel('Intensity (0-10)', fontsize=12, fontweight='bold', color='#334155')
    ax.set_title('Raw Acoustic Features Analysis', fontsize=16, fontweight='bold', color='#1e293b', pad=20)
    
    # Format time axis with proper time format
    def format_time(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes}:{seconds:02d}'
    
    formatter = matplotlib.ticker.FuncFormatter(format_time)
    ax.xaxis.set_major_formatter(formatter)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    
    # Grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#cbd5e1')
    
    # Add feature description annotations at notable points
    peak_points = {
        'pitch': (timestamps[np.argmax(pitches)], max(pitches)),
        'volume': (timestamps[np.argmax(volumes)], max(volumes)),
        'speech_rate': (timestamps[np.argmax(speech_rates)], max(speech_rates)),
        'vocal_tension': (timestamps[np.argmax(vocal_tensions)], max(vocal_tensions))
    }
    
    # Add annotations for peak points
    for feature, (x, y) in peak_points.items():
        ax.annotate(
            f'Peak {feature.replace("_", " ").title()}: {y}',
            xy=(x, y),
            xytext=(10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='#64748b'),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            fontsize=9
        )
    
    # Legend with custom styling
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4,
                      frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Set background color
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save chart
    chart_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    chart_filename = f"acoustic_features_{uuid.uuid4()}.png"
    chart_path = os.path.join(chart_dir, chart_filename)
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return chart_filename




def analyze_with_profile(media_path, profile_text, task_type, media_type):
    """Analyze emotions considering the profile information with enhanced modality-specific outputs"""
    try:
        # Format the profile information for the AI
        profile_prompt = ""
        if profile_text and len(profile_text.strip()) > 0:
            profile_prompt = f"""
            IMPORTANT CONTEXT - INDIVIDUAL PROFILE:
            {profile_text}
            
            Use this profile information to enhance your analysis, taking into account the individual's background, 
            psychological history, and personal context. This may help explain certain emotional patterns or responses.
            """
        
        # Get media duration and extract frames if needed
        frames = []
        frame_paths = []
        frame_times = []
        if media_type == "video":
            duration = get_video_duration(media_path)
            # Extract frames for more detailed analysis
            frames, frame_paths, frame_times, _ = extract_video_frames(media_path)
        else:  # audio
            duration = get_audio_duration(media_path)
        
        # Customize the prompt based on media type
        visual_prompt = ""
        audio_prompt = ""
        if media_type == "video":
            visual_prompt = """
            For video analysis, pay special attention to:
            - Facial expressions (micro-expressions, emotional changes)
            - Body posture and movement
            - Gestures and their meaning
            - Eye contact and gaze direction
            - Visual emotional cues in each frame
            """
        if media_type in ["video", "audio"]:
            audio_prompt = """
            For audio analysis, focus on:
            - Voice tone, pitch, and modulation
            - Speaking rate and rhythm
            - Pauses and hesitations
            - Voice tremors or tension
            - Vocal intensity and volume changes
            """
        
        # Structure prompt to get formatted output with timestamps and modality-specific analysis
        structured_prompt = f"""
        Task: Perform a detailed {media_type} analysis with timestamp information
        
        {profile_prompt}
        {visual_prompt}
        {audio_prompt}
        
        Instructions:
        Analyze this {media_type} and provide a STRUCTURED RESPONSE in valid JSON format with the following fields:
        
        1. "primary_emotion": The dominant emotion detected (string)
        
        2. "emotions": An array of detected emotions, each with:
           - "name": Emotion name (string)
           - "score": Confidence score from 1-10 (integer)
           - "justification": Brief explanation for this score (string)
        
        3. "timeline": An array of timestamp entries, each with:
           - "timestamp": Time in seconds (number)
           - "emotion": The emotion at this moment (string)
           - "description": Brief description of what's happening (string)
           - "key_sentence": Any important verbal content at this moment (string, can be empty)
        
        4. "behavioral_indicators": Key behavioral signals observed (array of strings)
        
        5. "analysis_summary": Brief overall assessment (string)
        
        6. "modality_analysis": Object containing:
           - "visual": Visual cue analysis - only for video (array of objects with "cue" and "interpretation")
           - "audio": Audio/vocal cue analysis (array of objects with "cue" and "interpretation")
           - "verbal": Speech content analysis (array of objects with "content" and "emotional_meaning")
        
        7. "raw_features": Object containing:
           - "audio_patterns": Array of objects with "timestamp", "pitch", "volume", "speech_rate", "vocal_tension" (all 0-10 scale)
           - "facial_landmarks": Array of objects with "timestamp", "expression_name", "confidence", "key_points" (coordinates)
           - "gesture_data": Array of objects with "timestamp", "gesture_type", "intensity", "meaning"
        
        The emotions to evaluate should include: happy, sad, angry, fearful, disgusted, surprised, contempt, neutral, anxious
        
        IMPORTANT: Don't just describe these features - extract and quantify them directly from the {media_type} data.
        For audio: Analyze the actual acoustic properties (pitch variations, volume levels, speech rate, vocal quality).
        For video: Detect and analyze actual facial landmarks, expressions, gestures, and posture changes.
        
        Your timestamps should be distributed throughout the media duration (approx. {int(duration)} seconds).
        
        Your final output must be a valid, parseable JSON object.
        """
                
        # For comprehensive analysis, use the most appropriate Gemini model
        model = "gemini-2.0-flash"
        
        # Create content list based on media type
        if media_type == "video":
            # Upload the video file
            media_file = genai_client.files.upload(file=media_path)
            
            # Wait for processing to complete
            while media_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                media_file = genai_client.files.get(name=media_file.name)

            if media_file.state.name == "FAILED":
                raise ValueError(f"Media processing failed: {media_file.state.name}")
            
            # Generate content with the video file
            response = genai_client.models.generate_content(
                model=model,
                contents=[structured_prompt, media_file],
                # Request structured output with schema
                config={
                    'response_mime_type': 'application/json'
                }
            )
            
            # Delete the file after use
            try:
                genai_client.files.delete(name=media_file.name)
            except Exception as e:
                print(f"Could not delete file: {e}")
                
        elif media_type == "audio":
            # Upload the audio file
            media_file = genai_client.files.upload(file=media_path)
            
            # Wait for processing to complete
            while media_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                media_file = genai_client.files.get(name=media_file.name)

            if media_file.state.name == "FAILED":
                raise ValueError(f"Media processing failed: {media_file.state.name}")
            
            # Generate content with the audio file
            response = genai_client.models.generate_content(
                model=model,
                contents=[structured_prompt, media_file],
                # Request structured output with schema
                config={
                    'response_mime_type': 'application/json'
                }
            )
            
            # Delete the file after use
            try:
                genai_client.files.delete(name=media_file.name)
            except Exception as e:
                print(f"Could not delete file: {e}")
        
        # Extract JSON from response
        result_text = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        # Remove any markdown code block indicators
        result_text = re.sub(r'```json', '', result_text)
        result_text = re.sub(r'```', '', result_text)
        result_text = result_text.strip()
        
        try:
            # Parse the JSON
            result_json = json.loads(result_text)
            
            # Add media information
            result_json['media_type'] = media_type
            result_json['duration'] = duration
            result_json['task_type'] = task_type
            result_json['has_profile'] = bool(profile_text and len(profile_text.strip()) > 0)
            
            # If we have timeline data and frame paths, perform DeepFace analysis
            # If we have timeline data and frame paths, perform MediaPipe face detection
            if 'timeline' in result_json and result_json['timeline'] and media_type == 'video':
                facial_landmarks_data = []
                
                # Extract more frames to catch more facial expressions
                additional_frames, additional_frame_paths, additional_frame_times, _ = extract_video_frames(media_path, num_frames=16)
                combined_frame_paths = frame_paths + additional_frame_paths
                combined_frame_times = frame_times + additional_frame_times
                
                # Process each timeline entry with MediaPipe
                for entry in result_json['timeline']:
                    timestamp = entry['timestamp']
                    emotion = entry['emotion']
                    
                    # Find the closest frame to this timestamp
                    closest_frame_index = 0
                    min_diff = float('inf')
                    for i, frame_time in enumerate(combined_frame_times):
                        diff = abs(frame_time - timestamp)
                        if diff < min_diff:
                            min_diff = diff
                            closest_frame_index = i
                    
                    if closest_frame_index < len(combined_frame_paths):
                        # Analyze the face in this frame using MediaPipe
                        landmarks_data = detect_faces_with_mediapipe(
                            combined_frame_paths[closest_frame_index], 
                            timestamp, 
                            emotion
                        )
                        if landmarks_data:
                            facial_landmarks_data.append(landmarks_data)
                
                # Add the facial landmarks data to the result
                if facial_landmarks_data:
                    # Make sure raw_features exists
                    if 'raw_features' not in result_json:
                        result_json['raw_features'] = {}
                    
                    # Add facial landmarks data
                    result_json['raw_features']['facial_landmarks'] = facial_landmarks_data
                    
                    # Generate the facial landmarks chart
                    landmarks_chart = generate_facial_landmarks_chart(facial_landmarks_data)
                    if landmarks_chart:
                        result_json['facial_landmarks_chart'] = landmarks_chart
            
                        # Process gesture analysis for the Raw Features tab
            if 'timeline' in result_json and result_json['timeline'] and media_type == 'video':
                gesture_data = []
                
                # Process every other timeline entry to get a variety of gestures
                for i, entry in enumerate(result_json['timeline']):
                    # Process every other entry to reduce duplication
                    if i % 2 == 0:
                        timestamp = entry['timestamp']
                        emotion = entry['emotion']
                        
                        # Find the closest frame to this timestamp
                        closest_frame_index = 0
                        min_diff = float('inf')
                        for j, frame_time in enumerate(frame_times):
                            diff = abs(frame_time - timestamp)
                            if diff < min_diff:
                                min_diff = diff
                                closest_frame_index = j
                        
                        if closest_frame_index < len(frame_paths):
                            # Analyze gestures in this frame
                            gesture_data_entry = detect_gestures_with_mediapipe(
                                frame_paths[closest_frame_index], 
                                timestamp, 
                                emotion
                            )
                            if gesture_data_entry:
                                gesture_data.append(gesture_data_entry)
                
                # Add the gesture data to the result
                if gesture_data:
                    # Make sure raw_features exists
                    if 'raw_features' not in result_json:
                        result_json['raw_features'] = {}
                    
                    # Add gesture data
                    result_json['raw_features']['gesture_data'] = gesture_data
                    
                    # Generate the gesture analysis chart
                    gesture_chart = generate_gesture_analysis_chart(gesture_data)
                    if gesture_chart:
                        result_json['gesture_analysis_chart'] = gesture_chart
                        
            # Generate enhanced emotion intensity chart
            if 'emotions' in result_json and result_json['emotions']:
                intensity_chart = generate_emotion_intensity_chart(result_json['emotions'])
                result_json['intensity_chart'] = intensity_chart
            
            # Generate enhanced timeline chart
            if 'timeline' in result_json and result_json['timeline']:
                timestamps = [entry['timestamp'] for entry in result_json['timeline']]
                emotions = [entry['emotion'] for entry in result_json['timeline']]
                timeline_chart = generate_emotion_timeline_chart(timestamps, emotions, duration)
                result_json['timeline_chart'] = timeline_chart
            
            # Generate additional visualizations for each analysis type
            if 'modality_analysis' in result_json:
                # Visual analysis chart (facial expression radar)
                if media_type == 'video' and result_json['modality_analysis'].get('visual'):
                    facial_chart = generate_facial_expression_radar(result_json['modality_analysis']['visual'])
                    if facial_chart:
                        result_json['facial_expression_chart'] = facial_chart
                        
                    # Generate facial landmarks visualization
                    # In analyze_with_profile function, replace the facial landmarks processing section:

                # Only process facial landmarks if not already processed
                if 'raw_features' not in result_json or 'facial_landmarks' not in result_json['raw_features']:
                    if 'timeline' in result_json and result_json['timeline'] and media_type == 'video':
                        facial_landmarks_data = []
                        
                        # Extract more frames to catch more facial expressions
                        additional_frames, additional_frame_paths, additional_frame_times, _ = extract_video_frames(media_path, num_frames=16)
                        combined_frame_paths = frame_paths + additional_frame_paths
                        combined_frame_times = frame_times + additional_frame_times
                        
                        # Process each timeline entry with MediaPipe
                        for entry in result_json['timeline']:
                            timestamp = entry['timestamp']
                            emotion = entry['emotion']
                            
                            # Find the closest frame to this timestamp
                            closest_frame_index = 0
                            min_diff = float('inf')
                            for i, frame_time in enumerate(combined_frame_times):
                                diff = abs(frame_time - timestamp)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_frame_index = i
                            
                            if closest_frame_index < len(combined_frame_paths):
                                # Analyze the face in this frame using MediaPipe
                                landmarks_data = detect_faces_with_mediapipe(
                                    combined_frame_paths[closest_frame_index], 
                                    timestamp, 
                                    emotion
                                )
                                if landmarks_data:
                                    facial_landmarks_data.append(landmarks_data)
                        
                        # Add the facial landmarks data to the result
                        if facial_landmarks_data:
                            # Make sure raw_features exists
                            if 'raw_features' not in result_json:
                                result_json['raw_features'] = {}
                            
                            # Add facial landmarks data
                            result_json['raw_features']['facial_landmarks'] = facial_landmarks_data
                            
                            # Generate the facial landmarks chart
                            landmarks_chart = generate_facial_landmarks_chart(facial_landmarks_data)
                            if landmarks_chart:
                                result_json['facial_landmarks_chart'] = landmarks_chart
                # Audio analysis chart (vocal pattern flow)
                if result_json['modality_analysis'].get('audio'):
                    vocal_chart = generate_vocal_pattern_chart(
                        result_json['modality_analysis']['audio'], 
                        result_json.get('timeline', [])
                    )
                    if vocal_chart:
                        result_json['vocal_pattern_chart'] = vocal_chart
                        
                    # Generate raw acoustic features chart if available
                    if 'raw_features' in result_json and 'audio_patterns' in result_json['raw_features']:
                        acoustic_chart = generate_raw_acoustic_features_chart(result_json['raw_features']['audio_patterns'])
                        if acoustic_chart:
                            result_json['acoustic_features_chart'] = acoustic_chart

                # Verbal content chart (sentiment flow)
                if result_json['modality_analysis'].get('verbal'):
                    sentiment_chart = generate_verbal_sentiment_flow(
                        result_json['modality_analysis']['verbal'],
                        result_json.get('timeline', [])
                    )
                    if sentiment_chart:
                        result_json['verbal_sentiment_chart'] = sentiment_chart
            
            return result_json
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response text: {result_text}")
            # Create a fallback JSON structure
            fallback_json = {
                "primary_emotion": "unknown",
                "emotions": [
                    {"name": "error", "score": 0, "justification": f"Failed to parse response: {str(e)}"}
                ],
                "timeline": [
                    {"timestamp": 0, "emotion": "unknown", "description": "Analysis error", "key_sentence": ""}
                ],
                "behavioral_indicators": ["Error in analysis"],
                "analysis_summary": f"The analysis failed to produce structured results. Raw response: {result_text[:500]}...",
                "modality_analysis": {
                    "visual": [],
                    "audio": [],
                    "verbal": []
                },
                "media_type": media_type,
                "duration": duration,
                "task_type": task_type,
                "has_profile": bool(profile_text and len(profile_text.strip()) > 0)
            }
            return fallback_json
            
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        error_json = {
            "primary_emotion": "error",
            "emotions": [
                {"name": "error", "score": 0, "justification": f"Analysis error: {str(e)}"}
            ],
            "timeline": [
                {"timestamp": 0, "emotion": "unknown", "description": "Analysis error", "key_sentence": ""}
            ],
            "behavioral_indicators": ["Analysis failed"],
            "analysis_summary": f"An error occurred during analysis: {str(e)}",
            "modality_analysis": {
                "visual": [],
                "audio": [],
                "verbal": []
            },
            "media_type": media_type,
            "duration": 0,
            "task_type": task_type,
            "has_profile": bool(profile_text and len(profile_text.strip()) > 0)
        }
        return error_json

def create_thumbnail_for_audio(audio_path):
    """Create a placeholder image for audio files"""
    thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails', f"audio_{uuid.uuid4()}.jpg")
    
    # Create a simple colored image with audio wave visualization
    img_width, img_height = 600, 300
    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 245  # Light gray background
    
    # Draw some simple audio wave patterns
    center_y = img_height // 2
    for x in range(0, img_width, 5):
        amplitude = 50 * np.sin(x * 0.05) * np.sin(x * 0.01)
        cv2.line(image, (x, center_y), (x, center_y + int(amplitude)), (41, 128, 185), 2)
        cv2.line(image, (x, center_y), (x, center_y - int(amplitude)), (41, 128, 185), 2)
    
    # Add audio icon
    cv2.circle(image, (img_width//2, img_height//2), 50, (52, 152, 219), -1)
    cv2.circle(image, (img_width//2, img_height//2), 20, (236, 240, 241), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Audio Analysis", (img_width//2 - 100, 30), font, 1, (44, 62, 80), 2, cv2.LINE_AA)
    
    cv2.imwrite(thumbnail_path, image)
    return thumbnail_path

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')
def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# In server.py, add this after your imports
import chat_functions

# Initialize the chat functionality during app startup
with app.app_context():
    chat_functions.initialize_chat()

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with the AI psychologist"""
    try:
        data = request.json
        user_message = data.get('message', '')
        analysis_id = data.get('analysis_id', '')
        
        if not user_message:
            return jsonify({'response': 'Please send a valid message.'}), 400
            
        # Check if we have a valid analysis in memory
        if not chat_functions.verify_analysis_session(analysis_id):
            return jsonify({'response': 'Sorry, I don\'t have any analysis data to refer to.'}), 400
        
        # Search for relevant context from the analysis
        contexts = chat_functions.search_relevant_context(user_message)
        
        # Generate response using Gemini
        bot_response = chat_functions.generate_chat_response(
            user_message, 
            contexts, 
            chat_functions.analysis_data.get('result', {}),
            genai_client  # Pass the Gemini client from server.py
        )
        
        return jsonify({'response': bot_response})
    
    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({'response': 'Sorry, I encountered an error processing your request.'}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle media upload and analysis"""
    # Determine media type
    media_type = request.form.get('media_type', 'video')
    task_type = request.form.get('task_type', 'emotion')
    
    # Get the appropriate file field name based on media type
    file_field = f"{media_type}_file"
    
    # Check if media file was provided
    if file_field not in request.files:
        flash(f'No {media_type} file provided')
        return redirect(url_for('index'))
    
    media_file = request.files[file_field]
    
    if media_file.filename == '':
        flash(f'No {media_type} file selected')
        return redirect(url_for('index'))
    
    # Check allowed extensions based on media type
    allowed_extensions = (app.config['ALLOWED_VIDEO_EXTENSIONS'] if media_type == 'video' 
                         else app.config['ALLOWED_AUDIO_EXTENSIONS'])
    
    if not allowed_file(media_file.filename, allowed_extensions):
        flash(f'Invalid {media_type} file type')
        return redirect(url_for('index'))
    
    # Save the media file
    media_filename = secure_filename(f"{media_type}_{uuid.uuid4()}_{media_file.filename}")
    media_path = os.path.join(app.config['UPLOAD_FOLDER'], media_filename)
    media_file.save(media_path)
    
    # Check for profile file
    profile_text = ""
    if 'profile_file' in request.files and request.files['profile_file'].filename != '':
        profile_file = request.files['profile_file']
        
        if allowed_file(profile_file.filename, app.config['ALLOWED_PROFILE_EXTENSIONS']):
            profile_filename = secure_filename(f"profile_{uuid.uuid4()}_{profile_file.filename}")
            profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'profiles', profile_filename)
            profile_file.save(profile_path)
            
            # Extract text from profile
            profile_text = extract_profile_text(profile_path)
        else:
            flash('Invalid profile file type')
    
    # Create thumbnail based on media type
    if media_type == 'video':
        try:
            frames, _, _, _ = extract_video_frames(media_path, num_frames=1)
            thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails', f"thumbnail_{uuid.uuid4()}.jpg")
            cv2.imwrite(thumbnail_path, frames[0])
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
            flash(f"Error processing video: {str(e)}")
            return redirect(url_for('index'))
    else:  # audio
        thumbnail_path = create_thumbnail_for_audio(media_path)
    
    # Generate a unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Analyze the media with profile information
    analysis_result = analyze_with_profile(media_path, profile_text, task_type, media_type)
    
    # Generate music recommendations based on analysis and profile
    music_recommendations = generate_music_recommendations(analysis_result, profile_text)
    
    # Add recommendations to the analysis result
    analysis_result['music_recommendations'] = music_recommendations
    
    # Create vectors for chat functionality
    chat_functions.create_analysis_vectors(analysis_result, analysis_id, profile_text)
    
    # Process thumbnail path for template
    thumbnail_filename = os.path.basename(thumbnail_path)
    
    return render_template(
        'result.html',
        result=analysis_result,
        media_path=url_for('uploaded_file', filename=media_filename),
        thumbnail=f"uploads/thumbnails/{thumbnail_filename}",
        task_type=task_type,
        media_type=media_type,
        has_profile=(profile_text != ""),
        charts_path="uploads/charts/",
        analysis_id=analysis_id  # Pass the analysis ID to the template
    )
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    if filename.startswith('charts/'):
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'charts'), 
                                  os.path.basename(filename))
    elif filename.startswith('thumbnails/'):
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'thumbnails'), 
                                  os.path.basename(filename))
    else:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    print("Emotion Analysis Flask application is running!")
    app.run(debug=True, host='0.0.0.0', port=5002)