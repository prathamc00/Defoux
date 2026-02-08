// DeepGuard AI - Deep Fake Detection System
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initNavigation();
    initFileUpload();
    initStatCounters();
});

// Particle Background System
function initParticles() {
    const container = document.getElementById('particles');
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (15 + Math.random() * 10) + 's';
        container.appendChild(particle);
    }
}

// Navigation
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    // Smooth scroll handling
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Active section tracking
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 150;
            if (window.scrollY >= sectionTop) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });
}

// File Upload System
// File Upload System
function initFileUpload() {
    const API_URL = 'http://localhost:8000';
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const previewVideo = document.getElementById('previewVideo');
    const removeBtn = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsContainer = document.getElementById('resultsContainer');

    let currentFile = null;

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Handle file selection
    function handleFile(file) {
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');

        if (!isImage && !isVideo) {
            showNotification('Please upload an image or video file', 'error');
            return;
        }

        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            if (isImage) {
                previewImage.src = e.target.result;
                previewImage.classList.add('active');
                previewVideo.classList.remove('active');
            } else {
                previewVideo.src = e.target.result;
                previewVideo.classList.add('active');
                previewImage.classList.remove('active');
            }

            uploadArea.style.display = 'none';
            previewArea.classList.add('active');
            analyzeBtn.classList.add('active');
            resultsContainer.classList.remove('active');
        };
        reader.readAsDataURL(file);
    }

    // Remove file
    removeBtn.addEventListener('click', () => {
        resetUpload();
    });

    function resetUpload() {
        currentFile = null;
        previewImage.src = '';
        previewVideo.src = '';
        previewImage.classList.remove('active');
        previewVideo.classList.remove('active');
        previewArea.classList.remove('active');
        analyzeBtn.classList.remove('active');
        resultsContainer.classList.remove('active');
        uploadArea.style.display = 'block';
        fileInput.value = '';
    }

    // Analyze button
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        analyzeBtn.classList.add('loading');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(`${API_URL}/api/v2/detect`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Analysis failed');
            }

            const data = await response.json();
            showResults(data);

        } catch (error) {
            console.error('Error:', error);
            showNotification(error.message, 'error');
        } finally {
            analyzeBtn.classList.remove('loading');
        }
    });

    function showResults(data) {
        resultsContainer.classList.add('active');
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

        const progressRing = document.getElementById('progressRing');
        const resultValue = document.getElementById('resultValue');
        const resultVerdict = document.getElementById('resultVerdict');

        // Calculate Authenticity Score
        let authenticityScore = 0;
        const isReal = data.result.toLowerCase() === 'real';

        // If Model says Real with 0.9 conf => 90% Authentic
        // If Model says Fake with 0.9 conf => 10% Authentic
        if (isReal) {
            authenticityScore = data.confidence * 100;
        } else {
            authenticityScore = (1 - data.confidence) * 100;
        }
        const finalScore = Math.round(authenticityScore);

        // Animate score counter
        animateValue(resultValue, 0, finalScore, 1500);

        // Animate progress ring
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (finalScore / 100) * circumference;
        progressRing.style.strokeDashoffset = offset;

        // Set verdict text and colors
        const resultSummary = document.getElementById('resultSummary');
        const resultReasoning = document.getElementById('resultReasoning');
        const reasoningText = document.getElementById('reasoningText');

        resultSummary.textContent = data.explanation.summary;

        if (data.explanation.confidence_reasoning) {
            reasoningText.textContent = data.explanation.confidence_reasoning;
            resultReasoning.style.display = 'block';
        } else {
            resultReasoning.style.display = 'none';
        }

        if (finalScore >= 80) {
            resultVerdict.textContent = 'Authentic Media';
            resultVerdict.className = 'result-verdict authentic';
            progressRing.style.stroke = '#22c55e';
            resultValue.style.color = '#22c55e';
        } else if (finalScore >= 50) {
            resultVerdict.textContent = 'Uncertain Determination';
            resultVerdict.className = 'result-verdict uncertain';
            progressRing.style.stroke = '#f59e0b';
            resultValue.style.color = '#f59e0b';
        } else {
            resultVerdict.textContent = 'Potential Deepfake';
            resultVerdict.className = 'result-verdict fake';
            progressRing.style.stroke = '#ef4444';
            resultValue.style.color = '#ef4444';
        }

        // Maps backend forensic evidence to UI bars
        const evidence = data.explanation.forensic_evidence;

        // Helper to set bar value. 
        // We invert "Anomaly Scores" (0-1, where 1 is bad) to "Integrity Scores" (0-100, where 100 is good).
        const updateBar = (barId, valId, score, suffix) => {
            const bar = document.getElementById(barId);
            const valEl = document.getElementById(valId);
            if (!bar || !valEl) return;

            // score is anomaly (0 good, 1 bad). So integrity = (1-score)*100
            let integrity = Math.round((1 - score) * 100);
            if (integrity < 0) integrity = 0;
            if (integrity > 100) integrity = 100;

            valEl.textContent = `${integrity}% ${suffix}`;
            setTimeout(() => {
                bar.style.width = integrity + '%';
            }, 300);
        };

        // 1. Frequency Analysis (FFT)
        // fft_score is high frequency ratio. 
        updateBar('fftBar', 'fftValue', evidence.fft_score, 'Normal');

        // 2. Color Consistency
        updateBar('colorBar', 'colorValue', evidence.color_score, 'Consistent');

        // 3. Noise Patterns
        // noise_score is variance. 
        // If variance is extreme (0 or 1), it's bad. If mid (0.5), it's good? 
        // For simplicity in this demo, we'll just treat the raw scalar as "Naturalness" 
        // or invert it if it aligns with anomaly logic.
        // rag_detection.py sets anomaly if <0.2 or >0.85. 
        // Let's use the boolean to decide a high/low score for visual clarity.
        const noiseIntegrity = evidence.noise_anomaly ? 0.3 : 0.95;
        //Pass 1-integrity as the 'score' argument because updateBar inverts it again.
        updateBar('noiseBar', 'noiseValue', 1 - noiseIntegrity, 'Natural');

        // 4. Compression Quality
        updateBar('compressionBar', 'compressionValue', evidence.compression_score, 'Quality');
    }
}

// Animate number counter
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        element.textContent = Math.floor(easeProgress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Hero Stats Counter Animation
function initStatCounters() {
    const stats = document.querySelectorAll('.stat-number');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const count = parseFloat(target.dataset.count);
                animateStatValue(target, 0, count, 2000);
                observer.unobserve(target);
            }
        });
    }, { threshold: 0.5 });

    stats.forEach(stat => observer.observe(stat));
}

function animateStatValue(element, start, end, duration) {
    let startTimestamp = null;
    const isDecimal = end % 1 !== 0;

    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = easeProgress * (end - start) + start;
        element.textContent = isDecimal ? current.toFixed(1) : Math.floor(current);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            element.textContent = isDecimal ? end.toFixed(1) : end;
        }
    };
    window.requestAnimationFrame(step);
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; padding: 16px 24px;
        background: ${type === 'error' ? '#ef4444' : '#00f0ff'}; color: ${type === 'error' ? '#fff' : '#0a0a0f'};
        border-radius: 12px; font-weight: 600; z-index: 9999;
        animation: fadeInUp 0.3s ease; box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'fadeIn 0.3s ease reverse';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Intersection Observer for scroll animations
const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };

const animateOnScroll = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.feature-card, .detail-card, .about-feature').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    animateOnScroll.observe(el);
});

// Tech grid animation
const techCells = document.querySelectorAll('.tech-cell');
setInterval(() => {
    techCells.forEach(cell => {
        if (Math.random() > 0.7) {
            cell.classList.toggle('active');
        }
    });
}, 1000);
