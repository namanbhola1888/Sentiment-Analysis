// Global variables
let currentFile = null;
let currentJobId = null;
let eventSource = null;
let startTime = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 3;
let progressStuckTimeout = null;
const PROGRESS_STUCK_THRESHOLD = 100000; // 60 seconds

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');

// Show loading overlay
function showLoading() {
    const loading = document.getElementById('loadingOverlay');
    if (!loading) {
        const overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        `;
        
        const spinner = document.createElement('div');
        spinner.style.cssText = `
            width: 50px;
            height: 50px;
            border: 5px solid #333;
            border-top: 5px solid #4fc3f7;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        `;
        
        const text = document.createElement('div');
        text.textContent = 'Loading Emotion Analysis...';
        text.style.cssText = `
            color: #4fc3f7;
            font-size: 18px;
            margin-top: 10px;
        `;
        
        overlay.appendChild(spinner);
        overlay.appendChild(text);
        document.body.appendChild(overlay);
        
        // Add CSS animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    } else {
        loading.style.display = 'flex';
    }
}

// Hide loading overlay
function hideLoading() {
    const loading = document.getElementById('loadingOverlay');
    if (loading) {
        loading.style.display = 'none';
    }
}

// Initialize with loading
document.addEventListener('DOMContentLoaded', () => {
    showLoading();
    
    // Check backend health
    checkBackend().finally(() => {
        hideLoading();
    });
    
    // Setup drag and drop
    setupDragAndDrop();
});

// Check backend status
async function checkBackend() {
    try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('Health check failed');
        
        const data = await response.json();
        const statusEl = document.getElementById('statusInfo');
        
        if (data.status === 'healthy') {
            statusEl.textContent = `Backend: Online`;
            statusEl.style.color = '#00c853';
        } else {
            statusEl.textContent = 'Backend: Offline';
            statusEl.style.color = '#ff5252';
        }
    } catch (error) {
        const statusEl = document.getElementById('statusInfo');
        statusEl.textContent = 'Backend: Connection Failed';
        statusEl.style.color = '#ff5252';
    }
}

async function checkSupabaseConnection() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        return data.supabase === true;
    } catch {
        return false;
    }
}

// Setup drag and drop
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.backgroundColor = '#1a1a1a';
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.style.backgroundColor = '';
        });
    });
    
    uploadArea.addEventListener('drop', handleDrop);
}

// Handle dropped files
function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFileSelect(files);
}

// Handle file selection
async function handleFileSelect(files) {
    if (!files.length) return;
    
    const file = files[0];
    
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showError('Please select a video file (MP4, AVI, MOV, MKV)');
        return;
    }
    
    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File size exceeds 50MB limit');
        return;
    }
    
    try {
        // Get video duration
        const duration = await getVideoDuration(file);
        if (duration > 25) {
            showError(`Video duration (${duration.toFixed(1)}s) exceeds 25 second limit`);
            return;
        }
        
        currentFile = file;
        displayFileInfo(file, duration);
        analyzeBtn.disabled = false;
        hideError();
    } catch (error) {
        showError('Could not read video file');
        console.error('File error:', error);
    }
}

// Get video duration
function getVideoDuration(file) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(video.src);
            resolve(video.duration);
        };
        
        video.onerror = () => {
            URL.revokeObjectURL(video.src);
            reject(new Error('Could not load video'));
        };
        
        video.src = URL.createObjectURL(file);
    });
}

// Display file information
function displayFileInfo(file, duration) {
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileDuration').textContent = `${duration.toFixed(1)}s`;
    fileInfo.style.display = 'block';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Clear selected file
function clearFile() {
    currentFile = null;
    fileInfo.style.display = 'none';
    analyzeBtn.disabled = true;
    fileInput.value = '';
    hideError();
}

// Show error message (generic for frontend)
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Show server error popup
function showServerError() {
    const existingPopup = document.querySelector('.server-error-popup');
    if (existingPopup) return;
    
    const popup = document.createElement('div');
    popup.className = 'server-error-popup';
    popup.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff5252;
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 300px;
    `;
    
    const message = document.createElement('div');
    message.innerHTML = `
        <strong style="display: block; margin-bottom: 5px;">⚠️ Server Error</strong>
        <div style="font-size: 12px;">Sorry, there was a server error. Please try again.</div>
    `;
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 5px;
        right: 10px;
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
    `;
    closeBtn.onclick = () => popup.remove();
    
    popup.appendChild(message);
    popup.appendChild(closeBtn);
    document.body.appendChild(popup);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (popup.parentNode) {
            popup.remove();
        }
    }, 5000);
    
    // Add CSS animation
    if (!document.querySelector('#popup-styles')) {
        const style = document.createElement('style');
        style.id = 'popup-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
}


// Show free instance error message
function showFreeInstanceError() {
    const existingPopup = document.querySelector('.free-instance-popup');
    if (existingPopup) return;
    
    const popup = document.createElement('div');
    popup.className = 'free-instance-popup';
    popup.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #ff9800;  // Orange warning color
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 350px;
    `;
    
    const message = document.createElement('div');
    message.innerHTML = `
        <strong style="display: block; margin-bottom: 8px;">⚠️ Free Instance Busy</strong>
        <div style="font-size: 13px; line-height: 1.4;">
            Processing takes longer than 4 minutes when all free instances are busy.<br>
            Please try again in 2-3 minutes or use a shorter video.
        </div>
    `;
    
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    closeBtn.style.cssText = `
        position: absolute;
        top: 5px;
        right: 10px;
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
    `;
    closeBtn.onclick = () => popup.remove();
    
    popup.appendChild(message);
    popup.appendChild(closeBtn);
    document.body.appendChild(popup);
    
    // Auto-remove after 8 seconds (longer message)
    setTimeout(() => {
        if (popup.parentNode) {
            popup.remove();
        }
    }, 8000);
    
    // Show upload section again
    document.querySelector('.upload-section').style.display = 'block';
    processingSection.style.display = 'none';
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
}

// Upload video for analysis
async function uploadVideo() {
    if (!currentFile) return;
    
    // Check Supabase connection first
    // const supabaseConnected = await checkSupabaseConnection();
    // if (!supabaseConnected) {
    //     showError('Database connection failed. Please refresh and try again.');
    //     return;
    // }

    const formData = new FormData();
    formData.append('video', currentFile);
    
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    hideError();
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        currentJobId = data.job_id;
        startProcessing();
        
    } catch (error) {
        showError(error.message);
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    }
}

// Start processing with SSE streaming
function startProcessing() {
    // Hide upload, show processing
    document.querySelector('.upload-section').style.display = 'none';
    processingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Reset processing UI
    resetProcessingUI();
    startTime = Date.now();
    reconnectAttempts = 0;
    
    // Start progress timer
    updateProgressTime();
    const timeInterval = setInterval(updateProgressTime, 1000);
    
    // Connect to SSE stream
    connectToSSE(timeInterval);
}

// Connect to SSE with retry logic
function connectToSSE(timeInterval) {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource(`/api/stream/${currentJobId}`);
    
    eventSource.onopen = () => {
        reconnectAttempts = 0;
    };
    
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            // ADD THIS CHECK - Specific for "Free instance busy" error
            if (data.error && data.error.includes('Free instance busy')) {
                showFreeInstanceError();
                clearInterval(timeInterval);
                eventSource.close();
                return;
            }

            if (data.error) {
                handleProcessingError(data.error);
                clearInterval(timeInterval);
                return;
            }
            
            updateProgressUI(data);
            
            if (data.status === 'completed') {
                clearInterval(timeInterval);
                setTimeout(() => showResults(), 1000);
            } else if (data.status === 'error') {
                clearInterval(timeInterval);
                handleProcessingError('Analysis failed');
            } else if (data.status === 'timeout') {
                clearInterval(timeInterval);
                handleProcessingError('Processing timeout');
            }
            
        } catch (error) {
            console.error('SSE parse error:', error);
        }
    };
    
    eventSource.onerror = (error) => {
        console.warn('SSE connection error:', error);
        
        if (eventSource.readyState === EventSource.CLOSED) {
            // Connection closed, attempt reconnect
            reconnectAttempts++;
            
            if (reconnectAttempts <= MAX_RECONNECT_ATTEMPTS) {
                document.getElementById('progressMessage').textContent = 
                    `Reconnecting... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`;
                
                setTimeout(() => {
                    if (currentJobId) {
                        connectToSSE(timeInterval);
                    }
                }, 2000 * reconnectAttempts); // Exponential backoff
            } else {
                // Max reconnection attempts reached
                clearInterval(timeInterval);
                handleProcessingError('Connection lost to server');
            }
        }
    };
}

// Reset processing UI
function resetProcessingUI() {
    document.getElementById('progressPercent').textContent = '0%';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressMessage').textContent = 'Initializing...';
    
    // Reset steps
    for (let i = 1; i <= 4; i++) {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active', 'completed');
        document.getElementById(`step${i}Status`).textContent = 'Pending';
    }
}

// Update progress time
function updateProgressTime() {
    if (!startTime) return;
    
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('progressTime').textContent = `${elapsed}s`;
}

// Update progress UI
function updateProgressUI(data) {
    const progress = data.progress || 0;
    const message = data.message || '';
    
    // Reset stuck timer if progress changed
    if (progress > parseInt(document.getElementById('progressPercent').textContent)) {
        if (progressStuckTimeout) {
            clearTimeout(progressStuckTimeout);
        }
        progressStuckTimeout = setTimeout(() => {
            if (parseInt(document.getElementById('progressPercent').textContent) < 20) {
                handleProcessingError('Processing stuck. Please try again.');
            }
        }, PROGRESS_STUCK_THRESHOLD);
    }
    
    // Update progress bar
    document.getElementById('progressPercent').textContent = `${progress}%`;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressMessage').textContent = message;
    
    // Update steps based on progress
    if (progress >= 10) updateStep(1, 'completed');
    if (progress >= 30) updateStep(2, progress >= 50 ? 'completed' : 'active');
    if (progress >= 60) updateStep(3, progress >= 75 ? 'completed' : 'active');
    if (progress >= 85) updateStep(4, progress >= 95 ? 'completed' : 'active');
}

// Update step status
function updateStep(stepNum, status) {
    const step = document.getElementById(`step${stepNum}`);
    const statusEl = document.getElementById(`step${stepNum}Status`);
    
    step.classList.remove('active', 'completed');
    step.classList.add(status);
    
    if (status === 'active') {
        statusEl.textContent = 'Processing';
        statusEl.style.color = '#2196f3';
    } else if (status === 'completed') {
        statusEl.textContent = 'Complete';
        statusEl.style.color = '#00c853';
    }
}

// Handle processing error
function handleProcessingError(message) {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
     if (progressStuckTimeout) {
        clearTimeout(progressStuckTimeout);
        progressStuckTimeout = null;
    }

    // Show server error popup (generic message)
    showServerError();
    
    // Show upload section again
    document.querySelector('.upload-section').style.display = 'block';
    processingSection.style.display = 'none';
    
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
}

// Show results
async function showResults() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    try {
        // Get results from API
        const response = await fetch(`/api/results/${currentJobId}`);
        const results = await response.json();
        
        if (!response.ok) {
            throw new Error(results.error || 'Failed to load results');
        }
        
        // Update UI with results
        updateResultsUI(results);
        
        // Show results section
        processingSection.style.display = 'none';
        resultsSection.style.display = 'block';
        
    } catch (error) {
        handleProcessingError(error.message);
    }
}

// Update results UI to display ALL data including transcript, emotions, and sentiment
function updateResultsUI(results) {
    console.log('Updating UI with complete results:', results);
    
    try {
        // 1. Set heatmap images from Base64 data
        const videoHeatmap = document.getElementById('videoHeatmap');
        const textHeatmap = document.getElementById('textHeatmap');
        
        // Load video heatmap (Emotion Heatmap)
        if (results.video_heatmap_data) {
            console.log('Loading video heatmap from Base64 data');
            
            // Create data URL from Base64
            const videoDataUrl = `data:image/png;base64,${results.video_heatmap_data}`;
            
            videoHeatmap.onload = () => {
                console.log('✅ Video heatmap loaded successfully');
                videoHeatmap.style.display = 'block';
                videoHeatmap.style.maxWidth = '100%';
                videoHeatmap.style.height = 'auto';
                videoHeatmap.style.borderRadius = '8px';
                videoHeatmap.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
            };
            
            videoHeatmap.onerror = (e) => {
                console.error('❌ Failed to load video heatmap:', e);
                videoHeatmap.style.display = 'none';
                showHeatmapPlaceholder(videoHeatmap, 'Emotion Heatmap Failed to Load');
            };
            
            videoHeatmap.src = videoDataUrl;
        } else {
            console.warn('No video heatmap data in results');
            showHeatmapPlaceholder(videoHeatmap, 'No Emotion Data Available');
        }
        
        // Load text heatmap (Sentiment Heatmap)
        if (results.text_heatmap_data) {
            console.log('Loading text heatmap from Base64 data');
            
            // Create data URL from Base64
            const textDataUrl = `data:image/png;base64,${results.text_heatmap_data}`;
            
            textHeatmap.onload = () => {
                console.log('✅ Text heatmap loaded successfully');
                textHeatmap.style.display = 'block';
                textHeatmap.style.maxWidth = '100%';
                textHeatmap.style.height = 'auto';
                textHeatmap.style.borderRadius = '8px';
                textHeatmap.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
            };
            
            textHeatmap.onerror = (e) => {
                console.error('❌ Failed to load text heatmap:', e);
                textHeatmap.style.display = 'none';
                showHeatmapPlaceholder(textHeatmap, 'Sentiment Heatmap Failed to Load');
            };
            
            textHeatmap.src = textDataUrl;
        } else {
            console.log('No text heatmap in results');
            showHeatmapPlaceholder(textHeatmap, 'No Sentiment Analysis Available');
        }
        
        // 2. Update summary stats
        const summaryStats = document.getElementById('summaryStats');
        if (summaryStats) {
            const duration = results.duration || 0;
            const frames = results.frames_analyzed || 0;
            const happiness = (results.emotion_summary?.happy || 0) * 100;
            const sentiment = results.sentiment?.compound || 0;
            
            summaryStats.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${duration.toFixed(1)}s</div>
                    <div class="stat-label">Duration</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${frames}</div>
                    <div class="stat-label">Frames Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${happiness.toFixed(0)}%</div>
                    <div class="stat-label">Avg Happiness</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${sentiment.toFixed(2)}</div>
                    <div class="stat-label">Sentiment Score</div>
                </div>
            `;
            console.log('✅ Summary stats updated');
        }
        
        // 3. Update transcript section
        const transcript = document.getElementById('transcript');
        if (transcript) {
            const transcriptText = results.transcript || 'No speech detected';
            
            // Format transcript with better styling
            transcript.innerHTML = `
                <div style="
                    background: rgba(255, 255, 255, 0.05);
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #4fc3f7;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #e0e0e0;
                    max-height: 200px;
                    overflow-y: auto;
                ">
                    ${transcriptText}
                </div>
            `;
            
            console.log('✅ Transcript updated:', transcriptText.length > 50 ? transcriptText.substring(0, 50) + '...' : transcriptText);
        }
        
        // 4. Update emotion distribution
        const emotionGrid = document.getElementById('emotionGrid');
        if (emotionGrid && results.emotion_summary) {
            const emotions = results.emotion_summary;
            let emotionHTML = '';
            
            // Emotion icons mapping
            const emotionIcons = {
                'angry': '😠',
                'disgust': '🤢',
                'fear': '😨',
                'happy': '😊',
                'sad': '😢',
                'surprise': '😲',
                'neutral': '😐'
            };
            
            // Emotion colors
            const emotionColors = {
                'angry': '#ef4444',
                'disgust': '#8b5cf6',
                'fear': '#7c3aed',
                'happy': '#10b981',
                'sad': '#3b82f6',
                'surprise': '#f59e0b',
                'neutral': '#64748b'
            };
            
            // Convert object to array and sort by value
            const emotionArray = Object.entries(emotions)
                .map(([name, value]) => ({
                    name,
                    value: value * 100, // Convert to percentage
                    icon: emotionIcons[name] || '❓',
                    color: emotionColors[name] || '#666'
                }))
                .sort((a, b) => b.value - a.value); // Sort descending
            
            // Generate emotion cards
            emotionArray.forEach(emotion => {
                emotionHTML += `
                    <div class="emotion-item" style="
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        transition: all 0.3s ease;
                    ">
                        <div style="font-size: 24px; margin-bottom: 8px;">${emotion.icon}</div>
                        <div class="emotion-name" style="
                            font-size: 12px;
                            color: #aaa;
                            margin-bottom: 5px;
                            text-transform: uppercase;
                            letter-spacing: 1px;
                        ">
                            ${emotion.name}
                        </div>
                        <div class="emotion-value" style="
                            font-size: 18px;
                            font-weight: bold;
                            color: ${emotion.color};
                        ">
                            ${emotion.value.toFixed(1)}%
                        </div>
                        <div style="
                            margin-top: 8px;
                            height: 6px;
                            background: rgba(255, 255, 255, 0.1);
                            border-radius: 3px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: ${emotion.value}%;
                                height: 100%;
                                background: ${emotion.color};
                                border-radius: 3px;
                            "></div>
                        </div>
                    </div>
                `;
            });
            
            emotionGrid.innerHTML = emotionHTML;
            console.log('✅ Emotion distribution updated with', emotionArray.length, 'emotions');
        } else {
            console.warn('No emotion data available');
            if (emotionGrid) {
                emotionGrid.innerHTML = `
                    <div style="
                        grid-column: 1 / -1;
                        text-align: center;
                        color: #888;
                        padding: 20px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                    ">
                        No emotion data available
                    </div>
                `;
            }
        }
        
        // 5. Update sentiment scores
        const sentimentScores = document.getElementById('sentimentScores');
        if (sentimentScores && results.sentiment) {
            const sentiment = results.sentiment;
            
            // Calculate percentages
            const negPercent = (sentiment.neg * 100).toFixed(1);
            const neuPercent = (sentiment.neu * 100).toFixed(1);
            const posPercent = (sentiment.pos * 100).toFixed(1);
            const compound = sentiment.compound;
            
            // Determine overall sentiment label and color
            let sentimentLabel = 'Neutral';
            let sentimentColor = '#f59e0b'; // Orange for neutral
            
            if (compound >= 0.05) {
                sentimentLabel = 'Positive';
                sentimentColor = '#10b981'; // Green for positive
            } else if (compound <= -0.05) {
                sentimentLabel = 'Negative';
                sentimentColor = '#ef4444'; // Red for negative
            }
            
            sentimentScores.innerHTML = `
                <div style="
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 15px;
                    margin-top: 10px;
                ">
                    <div class="sentiment-item" style="
                        background: rgba(239, 68, 68, 0.1);
                        border: 1px solid rgba(239, 68, 68, 0.3);
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div class="sentiment-label" style="
                            font-size: 12px;
                            color: #ef4444;
                            margin-bottom: 8px;
                            font-weight: 600;
                        ">
                            Negative
                        </div>
                        <div class="sentiment-value" style="
                            font-size: 20px;
                            font-weight: bold;
                            color: #ef4444;
                        ">
                            ${negPercent}%
                        </div>
                        <div style="
                            margin-top: 8px;
                            height: 6px;
                            background: rgba(239, 68, 68, 0.2);
                            border-radius: 3px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: ${negPercent}%;
                                height: 100%;
                                background: #ef4444;
                                border-radius: 3px;
                            "></div>
                        </div>
                    </div>
                    
                    <div class="sentiment-item" style="
                        background: rgba(245, 158, 11, 0.1);
                        border: 1px solid rgba(245, 158, 11, 0.3);
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div class="sentiment-label" style="
                            font-size: 12px;
                            color: #f59e0b;
                            margin-bottom: 8px;
                            font-weight: 600;
                        ">
                            Neutral
                        </div>
                        <div class="sentiment-value" style="
                            font-size: 20px;
                            font-weight: bold;
                            color: #f59e0b;
                        ">
                            ${neuPercent}%
                        </div>
                        <div style="
                            margin-top: 8px;
                            height: 6px;
                            background: rgba(245, 158, 11, 0.2);
                            border-radius: 3px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: ${neuPercent}%;
                                height: 100%;
                                background: #f59e0b;
                                border-radius: 3px;
                            "></div>
                        </div>
                    </div>
                    
                    <div class="sentiment-item" style="
                        background: rgba(16, 185, 129, 0.1);
                        border: 1px solid rgba(16, 185, 129, 0.3);
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div class="sentiment-label" style="
                            font-size: 12px;
                            color: #10b981;
                            margin-bottom: 8px;
                            font-weight: 600;
                        ">
                            Positive
                        </div>
                        <div class="sentiment-value" style="
                            font-size: 20px;
                            font-weight: bold;
                            color: #10b981;
                        ">
                            ${posPercent}%
                        </div>
                        <div style="
                            margin-top: 8px;
                            height: 6px;
                            background: rgba(16, 185, 129, 0.2);
                            border-radius: 3px;
                            overflow: hidden;
                        ">
                            <div style="
                                width: ${posPercent}%;
                                height: 100%;
                                background: #10b981;
                                border-radius: 3px;
                            "></div>
                        </div>
                    </div>
                    
                    <div class="sentiment-item" style="
                        background: rgba(79, 195, 247, 0.1);
                        border: 1px solid rgba(79, 195, 247, 0.3);
                        border-radius: 8px;
                        padding: 15px;
                        text-align: center;
                    ">
                        <div class="sentiment-label" style="
                            font-size: 12px;
                            color: #4fc3f7;
                            margin-bottom: 8px;
                            font-weight: 600;
                        ">
                            Overall Sentiment
                        </div>
                        <div class="sentiment-value" style="
                            font-size: 20px;
                            font-weight: bold;
                            color: ${sentimentColor};
                            margin-bottom: 5px;
                        ">
                            ${compound.toFixed(3)}
                        </div>
                        <div style="
                            font-size: 12px;
                            color: ${sentimentColor};
                            font-weight: 600;
                            padding: 4px 8px;
                            background: rgba(79, 195, 247, 0.2);
                            border-radius: 12px;
                            display: inline-block;
                        ">
                            ${sentimentLabel}
                        </div>
                    </div>
                </div>
            `;
            console.log('✅ Sentiment scores updated');
        } else {
            console.warn('No sentiment data available');
            if (sentimentScores) {
                sentimentScores.innerHTML = `
                    <div style="
                        text-align: center;
                        color: #888;
                        padding: 20px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                    ">
                        No sentiment data available
                    </div>
                `;
            }
        }
        
        // 6. Update download buttons
        const downloadVideoBtn = document.querySelector('[onclick*="downloadHeatmap(\'video\')"]');
        const downloadTextBtn = document.querySelector('[onclick*="downloadHeatmap(\'text\')"]');
        
        if (downloadVideoBtn && results.video_heatmap_data) {
            downloadVideoBtn.style.display = 'inline-flex';
            downloadVideoBtn.innerHTML = '<i class="fas fa-download"></i> Download Emotion Heatmap';
        }
        
        if (downloadTextBtn && results.text_heatmap_data) {
            downloadTextBtn.style.display = 'inline-flex';
            downloadTextBtn.innerHTML = '<i class="fas fa-download"></i> Download Sentiment Heatmap';
        }
        
        console.log('✅ All UI sections updated successfully');
        
    } catch (error) {
        console.error('❌ Error updating results UI:', error);
        console.error('Error details:', error.stack);
        
        // Show error in UI
        showGenericError();
    }
}

// Helper function to show placeholder if heatmap fails to load
function showHeatmapPlaceholder(imgElement, message) {
    imgElement.style.display = 'block';
    imgElement.alt = message;
    imgElement.style.maxWidth = '100%';
    imgElement.style.height = 'auto';
    imgElement.style.borderRadius = '8px';
    
    // Create a simple SVG placeholder
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300">
        <rect width="100%" height="100%" fill="#1a1a1a"/>
        <rect width="90%" height="80%" x="5%" y="10%" fill="#222" rx="8" ry="8"/>
        <text x="50%" y="45%" font-family="Arial, sans-serif" font-size="16" fill="#666" text-anchor="middle">
            ${message}
        </text>
        <text x="50%" y="55%" font-family="Arial, sans-serif" font-size="12" fill="#888" text-anchor="middle">
            Try analyzing another video
        </text>
    </svg>`;
    
    imgElement.src = 'data:image/svg+xml;base64,' + btoa(svg);
}

// Function to download heatmap
function downloadHeatmap(type) {
    if (!currentJobId) return;
    
    try {
        const heatmapImg = type === 'video' ? 
            document.getElementById('videoHeatmap') : 
            document.getElementById('textHeatmap');
        
        if (!heatmapImg || !heatmapImg.src) {
            console.error('No heatmap image to download');
            return;
        }
        
        // Create a temporary link
        const link = document.createElement('a');
        link.href = heatmapImg.src;
        link.download = `emotion_analysis_${type}_heatmap_${currentJobId}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log(`✅ Downloading ${type} heatmap`);
    } catch (error) {
        console.error('Error downloading heatmap:', error);
    }
}


// Reset for new analysis
function resetAnalysis() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    currentFile = null;
    currentJobId = null;
    reconnectAttempts = 0;
    
    // Reset UI
    document.querySelector('.upload-section').style.display = 'block';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Clear file info
    clearFile();
    analyzeBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    
    // Reset file input
    fileInput.value = '';
}

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files);
});