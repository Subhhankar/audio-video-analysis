/**
 * Parses the raw analysis text to extract structured data
 * @param {string} rawText - The raw analysis text from the API
 * @returns {object} - Structured analysis data
 */
function parseAnalysisData(rawText) {
    // Initialize the structured data object
    const result = {
        primaryEmotion: { name: 'Neutral', confidence: 50, icon: '😐' },
        emotions: [],
        visualCues: [],
        audioCues: [],
        transcript: [],
        timeline: [],
        profileContext: null
    };

    if (!rawText) return result;

    // Normalize line breaks and remove extra whitespace
    const normalizedText = rawText.replace(/\r\n/g, '\n').replace(/\n+/g, '\n').trim();

    // Extract the sections from the raw text
    const sections = extractSections(normalizedText);

    // Process each section
    if (sections.transcript) {
        result.transcript = parseTranscript(sections.transcript);
    }

    if (sections.visualAnalysis) {
        result.visualCues = parseVisualCues(sections.visualAnalysis);
    }

    if (sections.audioAnalysis) {
        result.audioCues = parseAudioCues(sections.audioAnalysis);
    }

    if (sections.emotionAssessment) {
        const emotionData = parseEmotionAssessment(sections.emotionAssessment);
        result.emotions = emotionData.emotions;
        
        // Set primary emotion if available
        if (emotionData.primaryEmotion) {
            result.primaryEmotion = emotionData.primaryEmotion;
        }
    }

    if (sections.temporal || sections.motionTemporal) {
        result.timeline = parseTimeline(sections.temporal || sections.motionTemporal);
    }

    if (sections.profileContext) {
        result.profileContext = sections.profileContext;
    }

    // If emotions are still empty, try to extract them from the full text
    if (result.emotions.length === 0) {
        result.emotions = extractEmotionsFromText(normalizedText);
    }

    // Ensure we have at least a default emotion
    if (result.emotions.length === 0) {
        result.emotions = [
            { name: 'Neutral', score: 5, color: 'neutral', details: 'Default emotional state', cues: ['No strong emotions detected'] }
        ];
    }

    // Add emoji icons to emotions
    result.emotions = result.emotions.map(emotion => {
        return { ...emotion, icon: getEmotionIcon(emotion.name) };
    });

    // Make sure the primary emotion has an icon
    result.primaryEmotion.icon = getEmotionIcon(result.primaryEmotion.name);

    return result;
}

/**
 * Extracts sections from the raw analysis text
 * @param {string} text - The normalized raw text
 * @returns {object} - Object containing the extracted sections
 */
function extractSections(text) {
    const sections = {};
    
    // Define section patterns
    const sectionPatterns = [
        { name: 'transcript', pattern: /TRANSCRIPT:([^]*)(?=VISUAL ANALYSIS:|AUDIO ANALYSIS:|$)/i },
        { name: 'visualAnalysis', pattern: /VISUAL ANALYSIS:([^]*)(?=AUDIO ANALYSIS:|MOTION & TEMPORAL PATTERNS:|TEMPORAL PATTERNS:|INTEGRATED EMOTION ASSESSMENT:|$)/i },
        { name: 'audioAnalysis', pattern: /AUDIO ANALYSIS:([^]*)(?=MOTION & TEMPORAL PATTERNS:|TEMPORAL PATTERNS:|INTEGRATED EMOTION ASSESSMENT:|$)/i },
        { name: 'motionTemporal', pattern: /MOTION & TEMPORAL PATTERNS:([^]*)(?=INTEGRATED EMOTION ASSESSMENT:|SUMMARY:|$)/i },
        { name: 'temporal', pattern: /TEMPORAL PATTERNS:([^]*)(?=INTEGRATED EMOTION ASSESSMENT:|SUMMARY:|$)/i },
        { name: 'emotionAssessment', pattern: /INTEGRATED EMOTION ASSESSMENT:([^]*)(?=SUMMARY:|PROFILE CONTEXT:|$)/i },
        { name: 'summary', pattern: /SUMMARY:([^]*)(?=PROFILE CONTEXT:|$)/i },
        { name: 'profileContext', pattern: /PROFILE CONTEXT:([^]*)/i }
    ];
    
    // Extract each section
    sectionPatterns.forEach(({ name, pattern }) => {
        const match = text.match(pattern);
        if (match && match[1]) {
            sections[name] = match[1].trim();
        }
    });
    
    return sections;
}

/**
 * Parses transcript text into structured data
 * @param {string} transcriptText - The transcript section text
 * @returns {Array} - Array of transcript entries
 */
function parseTranscript(transcriptText) {
    const transcript = [];
    
    // Pattern to match timestamp and text
    const entries = transcriptText.split('\n').filter(line => line.trim());
    
    entries.forEach(entry => {
        // Try to match timestamp formats (00:00, 0:00, [00:00], etc.)
        const timeMatch = entry.match(/\[?(\d+:\d+)\]?|^(\d+:\d+)/);
        
        if (timeMatch) {
            const time = timeMatch[1] || timeMatch[2];
            let text = entry.replace(timeMatch[0], '').trim();
            let emotion = 'Neutral';
            
            // Try to extract emotion if it's labeled in brackets or parentheses
            const emotionMatch = text.match(/\[(.*?)\]|\((.*?)\)/);
            if (emotionMatch) {
                emotion = emotionMatch[1] || emotionMatch[2];
                text = text.replace(emotionMatch[0], '').trim();
            }
            
            transcript.push({ time, text, emotion });
        } else {
            // Handle lines without timestamps (might be continuation of previous)
            if (transcript.length > 0) {
                const last = transcript[transcript.length - 1];
                last.text += ' ' + entry.trim();
            }
        }
    });
    
    return transcript;
}

/**
 * Parses visual cues text into structured data
 * @param {string} visualText - The visual analysis section text
 * @returns {Array} - Array of visual cue objects
 */
function parseVisualCues(visualText) {
    const visualCues = [];
    const paragraphs = visualText.split('\n').filter(line => line.trim());
    
    // Group paragraphs into categories
    let currentTitle = 'Facial Expressions';
    let currentContent = '';
    
    paragraphs.forEach(paragraph => {
        // Check if this paragraph is a new title
        const isTitleLine = /^[A-Z][A-Za-z\s]+:/.test(paragraph) || // "Category:" format
                           /^[•\-\*] [A-Z][A-Za-z\s]+:/.test(paragraph); // "• Category:" format
        
        if (isTitleLine) {
            // Save previous category if exists
            if (currentContent) {
                visualCues.push({ title: currentTitle, content: currentContent.trim() });
                currentContent = '';
            }
            
            // Extract new title
            currentTitle = paragraph.replace(/^[•\-\*] /, '').replace(/:$/, '').trim();
        } else {
            // Add to current content
            currentContent += paragraph + ' ';
        }
    });
    
    // Add the last category
    if (currentContent) {
        visualCues.push({ title: currentTitle, content: currentContent.trim() });
    }
    
    // Default categories if none were extracted
    if (visualCues.length === 0) {
        const defaultCategories = ['Facial Expressions', 'Body Language', 'Micro-expressions'];
        let remainingText = visualText;
        
        defaultCategories.forEach(category => {
            const chunk = remainingText.split('\n\n')[0];
            visualCues.push({ title: category, content: chunk || 'No specific details provided.' });
            remainingText = remainingText.replace(chunk, '').trim();
        });
    }
    
    return visualCues;
}

/**
 * Parses audio cues text into structured data
 * @param {string} audioText - The audio analysis section text
 * @returns {Array} - Array of audio cue objects
 */
function parseAudioCues(audioText) {
    const audioCues = [];
    const paragraphs = audioText.split('\n').filter(line => line.trim());
    
    // Logic similar to parseVisualCues
    let currentTitle = 'Voice Tone';
    let currentContent = '';
    
    paragraphs.forEach(paragraph => {
        const isTitleLine = /^[A-Z][A-Za-z\s]+:/.test(paragraph) || 
                           /^[•\-\*] [A-Z][A-Za-z\s]+:/.test(paragraph);
        
        if (isTitleLine) {
            if (currentContent) {
                audioCues.push({ title: currentTitle, content: currentContent.trim() });
                currentContent = '';
            }
            
            currentTitle = paragraph.replace(/^[•\-\*] /, '').replace(/:$/, '').trim();
        } else {
            currentContent += paragraph + ' ';
        }
    });
    
    if (currentContent) {
        audioCues.push({ title: currentTitle, content: currentContent.trim() });
    }
    
    // Default categories if none were extracted
    if (audioCues.length === 0) {
        const defaultCategories = ['Voice Tone', 'Speech Patterns', 'Vocal Indicators'];
        let remainingText = audioText;
        
        defaultCategories.forEach(category => {
            const chunk = remainingText.split('\n\n')[0];
            audioCues.push({ title: category, content: chunk || 'No specific details provided.' });
            remainingText = remainingText.replace(chunk, '').trim();
        });
    }
    
    return audioCues;
}

/**
 * Parses emotion assessment text into structured data
 * @param {string} assessmentText - The emotion assessment section text
 * @returns {object} - Object containing emotions and primary emotion
 */
function parseEmotionAssessment(assessmentText) {
    const result = {
        primaryEmotion: null,
        emotions: []
    };
    
    // Extract emotion scores (Emotion: N/10 format)
    const scorePattern = /([A-Za-z]+):\s*(\d+(?:\.\d+)?)\s*\/\s*10/g;
    let match;
    
    while ((match = scorePattern.exec(assessmentText)) !== null) {
        const name = match[1].trim();
        const score = parseFloat(match[2]);
        result.emotions.push({
            name: name,
            score: score,
            color: getEmotionColor(name),
            details: extractEmotionDetails(assessmentText, name),
            cues: extractEmotionCues(assessmentText, name)
        });
    }
    
    // If no scores found, try other formats
    if (result.emotions.length === 0) {
        // Try extracting primary/secondary emotions
        const primaryMatch = assessmentText.match(/primary(?:\s+emotion)?(?:\s+is)?(?:\s+identified\s+as)?[:\s]+([A-Za-z]+)/i);
        const secondaryMatch = assessmentText.match(/secondary(?:\s+emotions?)?(?:\s+include)?[:\s]+([A-Za-z,\s]+)/i);
        
        if (primaryMatch) {
            const name = primaryMatch[1].trim();
            result.emotions.push({
                name: name,
                score: 8.5, // Estimated score for primary
                color: getEmotionColor(name),
                details: extractEmotionDetails(assessmentText, name),
                cues: extractEmotionCues(assessmentText, name)
            });
        }
        
        if (secondaryMatch) {
            const secondaryEmotions = secondaryMatch[1].split(/,|\sand\s/).map(e => e.trim());
            secondaryEmotions.forEach((name, idx) => {
                if (name && name !== '') {
                    result.emotions.push({
                        name: name,
                        score: 7.0 - idx, // Decreasing scores for secondary emotions
                        color: getEmotionColor(name),
                        details: extractEmotionDetails(assessmentText, name),
                        cues: extractEmotionCues(assessmentText, name)
                    });
                }
            });
        }
    }
    
    // Sort emotions by score
    result.emotions.sort((a, b) => b.score - a.score);
    
    // Set primary emotion to the highest scoring one
    if (result.emotions.length > 0) {
        result.primaryEmotion = {
            name: result.emotions[0].name,
            confidence: Math.round(result.emotions[0].score * 10)
        };
    }
    
    return result;
}

/**
 * Extracts emotion details from text
 * @param {string} text - Text to search in
 * @param {string} emotionName - Name of the emotion
 * @returns {string} - Extracted details
 */
function extractEmotionDetails(text, emotionName) {
    // Look for sentences mentioning the emotion
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    const relevantSentences = sentences.filter(s => 
        s.toLowerCase().includes(emotionName.toLowerCase())
    );
    
    if (relevantSentences.length > 0) {
        return relevantSentences[0].trim();
    }
    
    // If no specific sentences found, return a general description
    return `The subject displays signs of ${emotionName.toLowerCase()} during the recording.`;
}

/**
 * Extracts emotion cues from text
 * @param {string} text - Text to search in
 * @param {string} emotionName - Name of the emotion
 * @returns {Array} - Array of cues
 */
function extractEmotionCues(text, emotionName) {
    const cues = [];
    const emotionRegex = new RegExp(`${emotionName}[^.!?]*?(?:through|via|by|in|with|including|such as)([^.!?]+)`, 'i');
    const match = text.match(emotionRegex);
    
    if (match && match[1]) {
        // Split by commas or 'and'
        const cueParts = match[1].split(/,|\sand\s/).map(p => p.trim());
        cueParts.forEach(part => {
            if (part && part.length > 3) { // Avoid very short parts
                cues.push(part);
            }
        });
    }
    
    // If no specific cues found, generate some based on the emotion
    if (cues.length === 0) {
        switch(emotionName.toLowerCase()) {
            case 'happy':
            case 'happiness':
                cues.push('Smiling', 'Upbeat vocal tone', 'Animated expressions');
                break;
            case 'sad':
            case 'sadness':
                cues.push('Downturned expression', 'Subdued tone', 'Reduced energy');
                break;
            case 'angry':
            case 'anger':
                cues.push('Furrowed brow', 'Tense facial muscles', 'Raised voice');
                break;
            case 'fear':
            case 'fearful':
            case 'afraid':
                cues.push('Widened eyes', 'Tense posture', 'Trembling voice');
                break;
            case 'surprise':
            case 'surprised':
                cues.push('Raised eyebrows', 'Widened eyes', 'Quick intake of breath');
                break;
            case 'disgust':
            case 'disgusted':
                cues.push('Wrinkled nose', 'Raised upper lip', 'Furrowed brow');
                break;
            case 'neutral':
                cues.push('Relaxed facial muscles', 'Even vocal tone', 'Minimal expressiveness');
                break;
            case 'worried':
            case 'worry':
            case 'concern':
            case 'concerned':
                cues.push('Furrowed brow', 'Compressed lips', 'Tense facial expression');
                break;
            default:
                cues.push('Facial expressions', 'Vocal tone', 'Body language');
        }
    }
    
    return cues;
}

/**
 * Parses timeline section into structured data
 * @param {string} timelineText - The timeline section text
 * @returns {Array} - Array of timeline events
 */
function parseTimeline(timelineText) {
    const timeline = [];
    const paragraphs = timelineText.split('\n').filter(line => line.trim());
    
    paragraphs.forEach(paragraph => {
        // Try to extract timestamp
        const timeMatch = paragraph.match(/\b(\d+:\d+)\b/);
        if (timeMatch) {
            const time = timeMatch[1];
            const content = paragraph.replace(timeMatch[0], '').trim();
            
            // Try to extract emotion
            let emotion = 'Neutral';
            const emotionMatch = content.match(/\b(happy|sad|angry|fear|surprise|disgust|neutral|worried)\b/i);
            if (emotionMatch) {
                emotion = emotionMatch[1].charAt(0).toUpperCase() + emotionMatch[1].slice(1).toLowerCase();
            }
            
            timeline.push({
                time: time,
                emotion: emotion,
                description: content
            });
        }
    });
    
    return timeline;
}

/**
 * Extracts emotions from the full text when structured sections aren't available
 * @param {string} fullText - The full analysis text
 * @returns {Array} - Array of emotion objects
 */
function extractEmotionsFromText(fullText) {
    const emotions = [];
    const commonEmotions = [
        'happy', 'happiness', 'joy', 'elation',
        'sad', 'sadness', 'sorrow', 'melancholy',
        'angry', 'anger', 'rage', 'frustration',
        'fear', 'fearful', 'afraid', 'anxiety',
        'surprise', 'surprised', 'shock', 'astonishment',
        'disgust', 'disgusted', 'repulsion',
        'neutral', 'calm', 'composed',
        'worried', 'worry', 'concern', 'concerned'
    ];
    
    // Create a regex to find passages discussing emotions
    const emotionRegex = new RegExp(`\\b(${commonEmotions.join('|')})\\b`, 'gi');
    const emotionMatches = {};
    
    let match;
    while ((match = emotionRegex.exec(fullText)) !== null) {
        const emotion = standardizeEmotionName(match[1]);
        emotionMatches[emotion] = (emotionMatches[emotion] || 0) + 1;
    }
    
    // Convert matches to scores
    Object.keys(emotionMatches).forEach(emotion => {
        const frequency = emotionMatches[emotion];
        let score = Math.min(frequency / 2, 10); // Rough heuristic based on frequency
        
        // Boost primary emotions mentioned early in the text
        if (fullText.indexOf(emotion) < fullText.length / 3) {
            score += 1;
        }
        
        emotions.push({
            name: emotion,
            score: Math.min(Math.round(score * 10) / 10, 10), // Round to 1 decimal, cap at 10
            color: getEmotionColor(emotion),
            details: extractEmotionDetails(fullText, emotion),
            cues: extractEmotionCues(fullText, emotion)
        });
    });
    
    // Sort by score
    emotions.sort((a, b) => b.score - a.score);
    
    return emotions;
}

/**
 * Standardizes emotion names to a consistent format
 * @param {string} emotion - Raw emotion name
 * @returns {string} - Standardized emotion name
 */
function standardizeEmotionName(emotion) {
    emotion = emotion.toLowerCase();
    
    // Map variants to standard names
    const emotionMap = {
        'happiness': 'Happy',
        'joy': 'Happy',
        'elation': 'Happy',
        'happy': 'Happy',
        
        'sadness': 'Sad',
        'sorrow': 'Sad',
        'melancholy': 'Sad',
        'sad': 'Sad',
        
        'anger': 'Angry',
        'rage': 'Angry',
        'frustration': 'Angry',
        'angry': 'Angry',
        
        'fear': 'Fear',
        'fearful': 'Fear',
        'afraid': 'Fear',
        'anxiety': 'Fear',
        
        'surprise': 'Surprise',
        'surprised': 'Surprise',
        'shock': 'Surprise',
        'astonishment': 'Surprise',
        
        'disgust': 'Disgust',
        'disgusted': 'Disgust',
        'repulsion': 'Disgust',
        
        'neutral': 'Neutral',
        'calm': 'Neutral',
        'composed': 'Neutral',
        
        'worry': 'Worried',
        'worried': 'Worried',
        'concern': 'Worried',
        'concerned': 'Worried'
    };
    
    return emotionMap[emotion] || 'Neutral';
}

/**
 * Gets the appropriate color for an emotion
 * @param {string} emotion - Emotion name
 * @returns {string} - Color class
 */
function getEmotionColor(emotion) {
    emotion = emotion.toLowerCase();
    
    const colorMap = {
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fear': 'fear',
        'surprise': 'surprise',
        'disgust': 'disgust',
        'neutral': 'neutral',
        'worried': 'worried'
    };
    
    // Handle emotion variants
    for (const [key, value] of Object.entries(colorMap)) {
        if (emotion.includes(key)) {
            return value;
        }
    }
    
    return 'neutral';
}

/**
 * Gets an emoji icon for an emotion
 * @param {string} emotion - Emotion name
 * @returns {string} - Emoji character
 */
function getEmotionIcon(emotion) {
    emotion = emotion.toLowerCase();
    
    const iconMap = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'worried': '😟'
    };
    
    // Handle emotion variants
    for (const [key, value] of Object.entries(iconMap)) {
        if (emotion.includes(key)) {
            return value;
        }
    }
    
    return '😐';
}

/**
 * Populates the UI with the structured analysis data
 * @param {object} data - Structured analysis data
 */
function populateAnalysisUI(data) {
    // Set primary emotion
    document.getElementById('primary-emotion-name').textContent = data.primaryEmotion.name;
    document.getElementById('primary-emotion-icon').textContent = data.primaryEmotion.icon;
    document.getElementById('primary-emotion-confidence').style.width = `${data.primaryEmotion.confidence}%`;
    document.getElementById('primary-emotion-confidence').textContent = `${data.primaryEmotion.confidence}%`;
    
    // Populate secondary emotions
    const secondaryEmotionsContainer = document.getElementById('secondary-emotions-container');
    secondaryEmotionsContainer.innerHTML = ''; // Clear existing content
    
    const secondaryEmotions = data.emotions.slice(1, 4); // Get 2nd-4th emotions
    secondaryEmotions.forEach(emotion => {
        const chip = document.createElement('div');
        chip.className = 'emotion-chip';
        chip.innerHTML = `${emotion.icon} ${emotion.name} <span class="score">${emotion.score}/10</span>`;
        secondaryEmotionsContainer.appendChild(chip);
    });
    
    // Populate detailed emotions
    const emotionsDetail = document.getElementById('emotions-detail');
    emotionsDetail.innerHTML = ''; // Clear existing content
    
    data.emotions.forEach(emotion => {
        const card = document.createElement('div');
        card.className = 'emotion-card';
        card.innerHTML = `
            <div class="emotion-header">
                <div class="emotion-title">
                    ${emotion.icon} ${emotion.name}
                </div>
                <div class="emotion-score">
                    <span class="score-value">${emotion.score}</span>
                    <span class="score-max">/10</span>
                </div>
            </div>
            <div class="emotion-bar">
                <div class="emotion-bar-fill color-${emotion.color}" style="width: ${emotion.score * 10}%"></div>
            </div>
            <div class="emotion-details">
                ${emotion.details}
            </div>
            <div class="emotion-cues">
                <ul class="cue-list">
                    ${emotion.cues.map(cue => `<li><i class="bi bi-check-circle"></i> ${cue}</li>`).join('')}
                </ul>
            </div>
        `;
        emotionsDetail.appendChild(card);
    });
    
    // Populate visual cues if available
    const visualCuesElement = document.getElementById('visual-cues');
    if (visualCuesElement && data.visualCues && data.visualCues.length > 0) {
        visualCuesElement.innerHTML = ''; // Clear existing content
        data.visualCues.forEach(cue => {
            const card = document.createElement('div');
            card.className = 'cue-card';
            card.innerHTML = `
                <div class="cue-card-title">
                    <i class="bi bi-eye"></i> ${cue.title}
                </div>
                <div class="cue-card-content">
                    ${cue.content}
                </div>
            `;
            visualCuesElement.appendChild(card);
        });
    }
    
    // Populate audio cues
    const audioCuesElement = document.getElementById('audio-cues');
    if (audioCuesElement && data.audioCues && data.audioCues.length > 0) {
        audioCuesElement.innerHTML = ''; // Clear existing content
        data.audioCues.forEach(cue => {
            const card = document.createElement('div');
            card.className = 'cue-card';
            card.innerHTML = `
                <div class="cue-card-title">
                    <i class="bi bi-volume-up"></i> ${cue.title}
                </div>
                <div class="cue-card-content">
                    ${cue.content}
                </div>
            `;
            audioCuesElement.appendChild(card);
        });
    }
    
    // Populate transcript
    const transcriptElement = document.getElementById('transcript');
    if (transcriptElement && data.transcript && data.transcript.length > 0) {
        transcriptElement.innerHTML = ''; // Clear existing content
        data.transcript.forEach(entry => {
            const div = document.createElement('div');
            div.className = 'transcript-entry';
            div.innerHTML = `
                <div class="transcript-time">${entry.time}</div>
                <div class="transcript-text">
                    ${entry.text}
                    <span class="transcript-emotion">${entry.emotion}</span>
                </div>
            `;
            transcriptElement.appendChild(div);
        });
    } else if (transcriptElement) {
        transcriptElement.innerHTML = '<p class="text-muted">No transcript available for this media.</p>';
    }
    
    // Populate timeline
    const timelineElement = document.getElementById('emotion-timeline');
    if (timelineElement && data.timeline && data.timeline.length > 0) {
        timelineElement.innerHTML = ''; // Clear existing content
        data.timeline.forEach(event => {
            const div = document.createElement('div');
            div.className = 'timeline-item';
            div.innerHTML = `
                <div class="timeline-time">${event.time}</div>
                <div class="timeline-content">
                    <span class="timeline-emotion">${getEmotionIcon(event.emotion)} ${event.emotion}</span>
                    <div class="timeline-description">${event.description}</div>
                </div>
            `;
            timelineElement.appendChild(div);
        });
    } else if (timelineElement) {
        timelineElement.innerHTML = '<p class="text-muted">No emotional timeline available for this media.</p>';
    }
    
    // Populate profile context if available
    const profileElement = document.getElementById('profile-context');
    if (profileElement && data.profileContext) {
        profileElement.innerHTML = `<div class="profile-content">${data.profileContext}</div>`;
    }
}

/**
 * Initializes the emotion chart
 * @param {Array} emotions - Array of emotion data
 */
function initEmotionChart(emotions) {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    
    // Only use top 6 emotions for the chart to avoid clutter
    const topEmotions = emotions.slice(0, 6);
    
    const emotionLabels = topEmotions.map(e => e.name);
    const emotionValues = topEmotions.map(e => e.score);
    const emotionColors = topEmotions.map(e => {
        switch(e.color) {
            case 'happy': return '#f1c40f';
            case 'sad': return '#3498db';
            case 'angry': return '#e74c3c';
            case 'fear': return '#9b59b6';
            case 'surprise': return '#e67e22';
            case 'disgust': return '#8e44ad';
            case 'neutral': return '#95a5a6';
            case 'worried': return '#d35400';
            default: return '#95a5a6';
        }
    });
    
    // Destroy any existing chart
    Chart.getChart(ctx.canvas)?.destroy();
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: emotionLabels,
            datasets: [{
                label: 'Emotion Intensity',
                data: emotionValues,
                backgroundColor: emotionColors,
                borderColor: emotionColors,
                borderWidth: 1,
                borderRadius: 5,
                maxBarThickness: 35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Score: ${context.parsed.y}/10`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 10,
                    ticks: {
                        stepSize: 2
                    },
                    title: {
                        display: true,
                        text: 'Intensity (0-10)'
                    }
                }
            }
        }
    });
}