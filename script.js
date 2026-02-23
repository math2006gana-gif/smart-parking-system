/**
 * Smart Parking AI - Modern Auto-Detection System
 */

// DOM Elements
const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const startCameraBtn = document.getElementById('startCameraBtn');
const manualCaptureBtn = document.getElementById('manualCaptureBtn');
const autoDetectToggle = document.getElementById('autoDetectToggle');
const systemStatus = document.getElementById('systemStatus');
const cameraLoading = document.getElementById('cameraLoading');
const countdownOverlay = document.getElementById('countdownOverlay');
const countdownNumber = document.getElementById('countdownNumber');
const processingState = document.getElementById('processingState');
const resultCard = document.getElementById('resultCard');
const errorCard = document.getElementById('errorCard');
const historyList = document.getElementById('historyList');

// Global Variables
let stream = null;
let autoDetectInterval = null;
let isProcessing = false;
let detectionHistory = [];
let totalDetections = 0;
let successfulDetections = 0;

// Auto-detection settings
const AUTO_DETECT_INTERVAL = 5000; // 5 seconds between auto-captures
const COUNTDOWN_DURATION = 3; // 3 second countdown

/**
 * Update System Status
 */
function updateSystemStatus(status, text) {
    const statusDot = systemStatus.querySelector('.status-dot');
    const statusText = systemStatus.querySelector('.status-text');
    
    statusDot.style.background = status === 'active' ? '#10b981' : 
                                  status === 'error' ? '#ef4444' : '#94a3b8';
    statusText.textContent = text;
}

/**
 * Show Toast Notification
 */
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Start Camera
 */
async function startCamera() {
    try {
        updateSystemStatus('loading', 'Starting Camera...');
        cameraLoading.classList.remove('hidden');
        
        // Check browser support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported in this browser');
        }
        
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({
            video: { 
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                facingMode: 'environment'
            }
        }).catch(() => {
            // Fallback to basic video
            return navigator.mediaDevices.getUserMedia({ video: true });
        });
        
        // Set video source
        videoElement.srcObject = stream;
        
        // Wait for video to load
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });
        
        // Hide loading, update UI
        cameraLoading.classList.add('hidden');
        startCameraBtn.disabled = true;
        startCameraBtn.textContent = '✓ Camera Active';
        manualCaptureBtn.disabled = false;
        
        updateSystemStatus('active', 'System Active');
        showToast('Camera started successfully!', 'success');
        
        // Start auto-detection if enabled
        if (autoDetectToggle.checked) {
            startAutoDetection();
        }
        
    } catch (error) {
        console.error('Camera error:', error);
        cameraLoading.classList.add('hidden');
        updateSystemStatus('error', 'Camera Error');
        
        let errorMsg = 'Failed to access camera. ';
        if (error.name === 'NotAllowedError') {
            errorMsg += 'Please allow camera permissions.';
        } else if (error.name === 'NotFoundError') {
            errorMsg += 'No camera found.';
        } else {
            errorMsg += error.message;
        }
        
        showToast(errorMsg, 'error');
    }
}

/**
 * Start Auto-Detection
 */
function startAutoDetection() {
    if (autoDetectInterval) return;
    
    console.log('Auto-detection enabled');
    showToast('Auto-detection enabled', 'success');
    
    autoDetectInterval = setInterval(() => {
        if (!isProcessing && stream) {
            captureWithCountdown();
        }
    }, AUTO_DETECT_INTERVAL);
}

/**
 * Stop Auto-Detection
 */
function stopAutoDetection() {
    if (autoDetectInterval) {
        clearInterval(autoDetectInterval);
        autoDetectInterval = null;
        console.log('Auto-detection disabled');
        showToast('Auto-detection disabled', 'success');
    }
}

/**
 * Capture with Countdown
 */
async function captureWithCountdown() {
    if (isProcessing) return;
    
    countdownOverlay.classList.add('active');
    
    for (let i = COUNTDOWN_DURATION; i > 0; i--) {
        countdownNumber.textContent = i;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    countdownOverlay.classList.remove('active');
    await captureAndProcess();
}

/**
 * Capture Photo
 */
function capturePhoto() {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    const context = canvasElement.getContext('2d');
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    return canvasElement.toDataURL('image/jpeg', 0.95);
}

/**
 * Capture and Process
 */
async function captureAndProcess() {
    if (isProcessing) {
        showToast('Already processing...', 'error');
        return;
    }
    
    isProcessing = true;
    totalDetections++;
    
    // Show processing state
    resultCard.style.display = 'none';
    errorCard.style.display = 'none';
    processingState.style.display = 'block';
    
    try {
        const base64Image = capturePhoto();
        
        const response = await fetch('/capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        processingState.style.display = 'none';
        
        if (data.success) {
            successfulDetections++;
            displayResult(data);
            addToHistory(data);
            updateStats();
            showToast(`Vehicle ${data.action === 'entry' ? 'entered' : 'exited'}: ${data.plate}`, 'success');
        } else {
            displayError(data.error);
            showToast('Detection failed', 'error');
        }
        
    } catch (error) {
        console.error('Capture error:', error);
        processingState.style.display = 'none';
        displayError('Network error: Could not connect to server');
        showToast('Connection error', 'error');
    } finally {
        isProcessing = false;
    }
}

/**
 * Display Result
 */
function displayResult(data) {
    resultCard.style.display = 'block';
    errorCard.style.display = 'none';
    
    // Plate number
    document.getElementById('plateNumber').textContent = data.plate;
    
    // Confidence
    const confidence = Math.round(data.confidence * 100);
    document.getElementById('confidenceValue').textContent = confidence + '%';
    document.getElementById('confidenceFill').style.width = confidence + '%';
    
    // Change color based on confidence
    const confidenceFill = document.getElementById('confidenceFill');
    if (confidence >= 80) {
        confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
    } else if (confidence >= 60) {
        confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
    } else {
        confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
    }
    
    // Action badge
    const actionBadge = document.getElementById('actionBadge');
    actionBadge.textContent = data.action.toUpperCase();
    actionBadge.className = `action-badge ${data.action}`;
    
    // Record ID
    document.getElementById('recordId').textContent = '#' + data.record_id;
    
    // Timestamp
    const now = new Date();
    document.getElementById('timestamp').textContent = now.toLocaleString();
    document.getElementById('resultTime').textContent = now.toLocaleTimeString();
    
    // Images
    document.getElementById('originalImage').src = `/uploads/${data.image_path}`;
    document.getElementById('plateImage').src = `/uploads/${data.roi_path}`;
}

/**
 * Display Error
 */
function displayError(message) {
    errorCard.style.display = 'block';
    resultCard.style.display = 'none';
    document.getElementById('errorMessage').textContent = message;
}

/**
 * Add to History
 */
function addToHistory(data) {
    const historyItem = {
        plate: data.plate,
        action: data.action,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString(),
        timestamp: Date.now()
    };
    
    detectionHistory.unshift(historyItem);
    
    // Keep only last 10
    if (detectionHistory.length > 10) {
        detectionHistory.pop();
    }
    
    renderHistory();
}

/**
 * Render History
 */
function renderHistory() {
    historyList.innerHTML = '';
    
    detectionHistory.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div>
                <div class="history-plate">${item.plate}</div>
                <div class="history-time">${item.time} • ${item.action.toUpperCase()}</div>
            </div>
            <div class="action-badge ${item.action}">${item.action}</div>
        `;
        historyList.appendChild(div);
    });
}

/**
 * Update Stats
 */
function updateStats() {
    document.getElementById('totalDetections').textContent = `Total: ${totalDetections}`;
    
    const successRate = totalDetections > 0 
        ? Math.round((successfulDetections / totalDetections) * 100) 
        : 100;
    document.getElementById('successRate').textContent = `Success: ${successRate}%`;
}

/**
 * Clear History
 */
function clearHistory() {
    detectionHistory = [];
    historyList.innerHTML = '<p style="text-align:center;color:var(--text-muted);">No detections yet</p>';
    showToast('History cleared', 'success');
}

// Event Listeners
startCameraBtn.addEventListener('click', startCamera);
manualCaptureBtn.addEventListener('click', () => captureWithCountdown());
document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);

// Auto-detection toggle
autoDetectToggle.addEventListener('change', (e) => {
    if (e.target.checked) {
        if (stream) {
            startAutoDetection();
        }
    } else {
        stopAutoDetection();
    }
});

// Auto-start camera on load (optional)
window.addEventListener('load', () => {
    console.log('Smart Parking AI loaded');
    updateSystemStatus('idle', 'Ready to Start');
    
    // Uncomment to auto-start camera
    // setTimeout(() => startCamera(), 1000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    stopAutoDetection();
});

console.log('✓ Smart Parking AI System Ready');