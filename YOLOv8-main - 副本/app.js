const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush'
];

const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1', '#FF6347', '#7B68EE',
  '#3CB371', '#FF69B4', '#00FA9A', '#FFD700', '#BA55D3', '#00BFFF'
];

const MODEL_URL = 'https://huggingface.co/spaces/schwenk/yolov8-demo/resolve/main/yolov8n.onnx';

let model = null;
let modelLoaded = false;
let currentImage = null;
let originalImageSize = { width: 0, height: 0 };

const particles = document.getElementById('particles');
const particlesCtx = particles.getContext('2d');
const uploadZone = document.getElementById('uploadZone');
const imageInput = document.getElementById('imageInput');
const fileInfo = document.getElementById('fileInfo');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');
const confThreshold = document.getElementById('confThreshold');
const iouThreshold = document.getElementById('iouThreshold');
const confValue = document.getElementById('confValue');
const iouValue = document.getElementById('iouValue');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultSection = document.getElementById('resultSection');
const resultCanvas = document.getElementById('resultCanvas');
const canvasPlaceholder = document.getElementById('canvasPlaceholder');
const totalDetections = document.getElementById('totalDetections');
const inferenceTime = document.getElementById('inferenceTime');
const imageDimensions = document.getElementById('imageDimensions');
const listContainer = document.getElementById('listContainer');
const downloadBtn = document.getElementById('downloadBtn');
const resetBtn = document.getElementById('resetBtn');
const toast = document.getElementById('toast');

function initParticles() {
  particles.width = window.innerWidth;
  particles.height = window.innerHeight;
  
  const particleCount = 80;
  const particleList = [];
  
  for (let i = 0; i < particleCount; i++) {
    particleList.push({
      x: Math.random() * particles.width,
      y: Math.random() * particles.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      radius: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.2
    });
  }
  
  function animate() {
    particlesCtx.clearRect(0, 0, particles.width, particles.height);
    
    particleList.forEach((p, i) => {
      p.x += p.vx;
      p.y += p.vy;
      
      if (p.x < 0 || p.x > particles.width) p.vx *= -1;
      if (p.y < 0 || p.y > particles.height) p.vy *= -1;
      
      particlesCtx.beginPath();
      particlesCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      particlesCtx.fillStyle = `rgba(16, 185, 129, ${p.opacity})`;
      particlesCtx.fill();
      
      particleList.slice(i + 1).forEach(p2 => {
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 150) {
          particlesCtx.beginPath();
          particlesCtx.moveTo(p.x, p.y);
          particlesCtx.lineTo(p2.x, p2.y);
          particlesCtx.strokeStyle = `rgba(16, 185, 129, ${0.15 * (1 - dist / 150)})`;
          particlesCtx.stroke();
        }
      });
    });
    
    requestAnimationFrame(animate);
  }
  
  animate();
  
  window.addEventListener('resize', () => {
    particles.width = window.innerWidth;
    particles.height = window.innerHeight;
  });
}

function showToast(message, type = 'info') {
  const toastMessage = toast.querySelector('.toast-message');
  toastMessage.textContent = message;
  toast.className = `toast show ${type}`;
  
  setTimeout(() => {
    toast.className = 'toast';
  }, 3000);
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function updateProgress(percent, text) {
  progressFill.style.width = percent + '%';
  progressText.textContent = text;
}

async function loadModel() {
  if (modelLoaded) return;
  
  try {
    updateProgress(10, '正在加载 ONNX Runtime...');
    
    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgl', 'wasm'],
      graphOptimizationLevel: 'all'
    });
    
    model = session;
    modelLoaded = true;
    updateProgress(100, '模型加载完成');
    showToast('模型加载成功！', 'success');
    
    setTimeout(() => {
      progressContainer.classList.remove('active');
    }, 1000);
    
  } catch (error) {
    console.error('模型加载失败:', error);
    showToast('模型加载失败，请刷新重试', 'error');
    updateProgress(0, '模型加载失败');
  }
}

function handleFileSelect(file) {
  if (!file) return;
  
  if (!file.type.startsWith('image/')) {
    showToast('请选择图片文件', 'error');
    return;
  }
  
  const reader = new FileReader();
  reader.onload = (e) => {
    currentImage = new Image();
    currentImage.onload = () => {
      originalImageSize = {
        width: currentImage.width,
        height: currentImage.height
      };
      
      previewImage.src = e.target.result;
      fileName.textContent = file.name;
      fileSize.textContent = formatFileSize(file.size);
      fileInfo.classList.add('active');
      detectBtn.disabled = false;
      
      showToast('图片加载成功', 'success');
    };
    currentImage.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function letterbox(image, targetSize = 640) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  const scale = Math.min(targetSize / image.width, targetSize / image.height);
  const newWidth = Math.round(image.width * scale);
  const newHeight = Math.round(image.height * scale);
  
  const padX = (targetSize - newWidth) / 2;
  const padY = (targetSize - newHeight) / 2;
  
  canvas.width = targetSize;
  canvas.height = targetSize;
  
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(image, padX, padY, newWidth, newHeight);
  
  return { canvas, scale, padX, padY };
}

function preprocessImage(image) {
  const { canvas, scale, padX, padY } = letterbox(image, 640);
  const ctx = canvas.getContext('2d');
  
  const imageData = ctx.getImageData(0, 0, 640, 640);
  const data = imageData.data;
  
  const float32Data = new Float32Array(1 * 3 * 640 * 640);
  
  for (let i = 0; i < 640 * 640; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    
    float32Data[i] = r;
    float32Data[640 * 640 + i] = g;
    float32Data[2 * 640 * 640 + i] = b;
  }
  
  return { tensor: float32Data, scale, padX, padY };
}

function nms(boxes, scores, iouThreshold) {
  const indices = [];
  const sortedIndices = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .map(item => item.index);
  
  while (sortedIndices.length > 0) {
    const current = sortedIndices.shift();
    indices.push(current);
    
    const remaining = [];
    for (const idx of sortedIndices) {
      const iou = calculateIoU(boxes[current], boxes[idx]);
      if (iou < iouThreshold) {
        remaining.push(idx);
      }
    }
    sortedIndices.length = 0;
    sortedIndices.push(...remaining);
  }
  
  return indices;
}

function calculateIoU(box1, box2) {
  const x1 = Math.max(box1.x, box2.x);
  const y1 = Math.max(box1.y, box2.y);
  const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
  const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
  
  if (x2 < x1 || y2 < y1) return 0;
  
  const intersection = (x2 - x1) * (y2 - y1);
  const area1 = box1.width * box1.height;
  const area2 = box2.width * box2.height;
  const union = area1 + area2 - intersection;
  
  return intersection / union;
}

function parseOutput(output, confThreshold, iouThreshold, scale, padX, padY) {
  const detections = [];
  const numDetections = output.dims[2];
  const data = output.data;
  
  const boxes = [];
  const scores = [];
  const classIds = [];
  
  for (let i = 0; i < numDetections; i++) {
    const centerX = data[i];
    const centerY = data[ numDetections + i];
    const width = data[2 * numDetections + i];
    const height = data[3 * numDetections + i];
    
    let maxScore = 0;
    let classId = 0;
    
    for (let c = 4; c < 84; c++) {
      const score = data[c * numDetections + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c - 4;
      }
    }
    
    if (maxScore > confThreshold) {
      const x = (centerX - padX) / scale;
      const y = (centerY - padY) / scale;
      const w = width / scale;
      const h = height / scale;
      
      boxes.push({
        x: x - w / 2,
        y: y - h / 2,
        width: w,
        height: h
      });
      scores.push(maxScore);
      classIds.push(classId);
    }
  }
  
  const keepIndices = nms(boxes, scores, iouThreshold);
  
  for (const idx of keepIndices) {
    detections.push({
      classId: classIds[idx],
      className: COCO_CLASSES[classIds[idx]] || 'unknown',
      confidence: scores[idx],
      bbox: boxes[idx],
      color: COLORS[classIds[idx] % COLORS.length]
    });
  }
  
  return detections;
}

function drawDetections(image, detections) {
  const canvas = resultCanvas;
  const ctx = canvas.getContext('2d');
  
  const maxWidth = 800;
  const scale = Math.min(1, maxWidth / image.width);
  
  canvas.width = image.width * scale;
  canvas.height = image.height * scale;
  
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  
  ctx.font = '14px "Noto Sans SC"';
  ctx.lineWidth = 2;
  
  for (const det of detections) {
    const { x, y, width, height } = det.bbox;
    const scaledX = x * scale;
    const scaledY = y * scale;
    const scaledW = width * scale;
    const scaledH = height * scale;
    
    ctx.strokeStyle = det.color;
    ctx.lineWidth = 2;
    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
    
    const label = `${det.className} ${(det.confidence * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(label).width;
    
    ctx.fillStyle = det.color;
    ctx.fillRect(scaledX, scaledY - 22, textWidth + 10, 22);
    
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(label, scaledX + 5, scaledY - 6);
  }
  
  canvasPlaceholder.classList.add('hidden');
}

function displayDetectionList(detections) {
  listContainer.innerHTML = '';
  
  if (detections.length === 0) {
    listContainer.innerHTML = '<p style="color: var(--text-muted); text-align: center;">未检测到目标</p>';
    return;
  }
  
  for (const det of detections) {
    const item = document.createElement('div');
    item.className = 'detection-item';
    item.style.borderLeftColor = det.color;
    item.innerHTML = `
      <div class="detection-color" style="background: ${det.color}"></div>
      <span class="detection-name">${det.className}</span>
      <span class="detection-conf">${(det.confidence * 100).toFixed(1)}%</span>
    `;
    listContainer.appendChild(item);
  }
}

async function runDetection() {
  if (!currentImage || !model) {
    showToast('请先上传图片', 'error');
    return;
  }
  
  detectBtn.classList.add('loading');
  detectBtn.querySelector('.btn-text').textContent = '检测中...';
  progressContainer.classList.add('active');
  
  const startTime = performance.now();
  
  try {
    updateProgress(20, '预处理图像...');
    const { tensor, scale, padX, padY } = preprocessImage(currentImage);
    
    updateProgress(40, '执行模型推理...');
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
    const results = await model.run({ images: inputTensor });
    
    updateProgress(70, '解析检测结果...');
    const output = results[Object.keys(results)[0]];
    const conf = parseFloat(confThreshold.value);
    const iou = parseFloat(iouThreshold.value);
    const detections = parseOutput(output, conf, iou, scale, padX, padY);
    
    updateProgress(90, '绘制结果...');
    drawDetections(currentImage, detections);
    displayDetectionList(detections);
    
    const endTime = performance.now();
    const inferenceMs = Math.round(endTime - startTime);
    
    totalDetections.textContent = detections.length;
    inferenceTime.textContent = inferenceMs;
    imageDimensions.textContent = `${originalImageSize.width}×${originalImageSize.height}`;
    
    resultSection.classList.add('active');
    resultSection.scrollIntoView({ behavior: 'smooth' });
    
    updateProgress(100, '检测完成');
    showToast(`检测完成！发现 ${detections.length} 个目标`, 'success');
    
  } catch (error) {
    console.error('检测失败:', error);
    showToast('检测失败: ' + error.message, 'error');
    updateProgress(0, '检测失败');
  } finally {
    detectBtn.classList.remove('loading');
    detectBtn.querySelector('.btn-text').textContent = '开始检测';
    
    setTimeout(() => {
      progressContainer.classList.remove('active');
    }, 1500);
  }
}

function downloadResult() {
  const canvas = resultCanvas;
  const link = document.createElement('a');
  link.download = 'detection_result.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
  showToast('图片已下载', 'success');
}

function resetDetection() {
  currentImage = null;
  fileInfo.classList.remove('active');
  detectBtn.disabled = true;
  resultSection.classList.remove('active');
  canvasPlaceholder.classList.remove('hidden');
  
  const ctx = resultCanvas.getContext('2d');
  ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
  
  listContainer.innerHTML = '';
  totalDetections.textContent = '0';
  inferenceTime.textContent = '0';
  imageDimensions.textContent = '-';
  
  document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
}

function init() {
  initParticles();
  
  uploadZone.addEventListener('click', () => imageInput.click());
  
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  
  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });
  
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  });
  
  imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  });
  
  removeBtn.addEventListener('click', () => {
    currentImage = null;
    fileInfo.classList.remove('active');
    detectBtn.disabled = true;
    imageInput.value = '';
  });
  
  confThreshold.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
  
  iouThreshold.addEventListener('input', (e) => {
    iouValue.textContent = parseFloat(e.target.value).toFixed(2);
  });
  
  detectBtn.addEventListener('click', async () => {
    if (!modelLoaded) {
      progressContainer.classList.add('active');
      await loadModel();
    }
    if (modelLoaded) {
      await runDetection();
    }
  });
  
  downloadBtn.addEventListener('click', downloadResult);
  resetBtn.addEventListener('click', resetDetection);
  
  showToast('欢迎使用 YOLO 目标检测系统', 'success');
}

document.addEventListener('DOMContentLoaded', init);
