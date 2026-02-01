const canvas = document.getElementById("sketchpad");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const pointsOutput = document.getElementById("pointsOutput");

let isDrawing = false;
let lastPoint = null;
let hasDrawnLine = false;
let points = [];
let pointsNorm = [];

function getPos(e) {
  // Get the canvas position in the page
  const rect = canvas.getBoundingClientRect();

  // Support both mouse and touch
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;

  // Coordinates relative to the top-left corner of the canvas
  return normPoints(clientX - rect.left, clientY - rect.top, rect.width, rect.height)
}

function normPoints(x, y, w, h) {
  return {
    x: x / w,
    y: y / h
  }
}

function deNormPoints(x, y, w, h) {
  return {
    x_dn: Math.round(x * w),
    y_dn: Math.round(y * h)
  }
}

function startDraw(e) {
  if (hasDrawnLine) return;

  isDrawing = true;
  lastPoint = null;
  points = []; // start fresh stroke
  pointsNorm = [];

  const rect = canvas.getBoundingClientRect();

  const { x, y } = getPos(e);
  ctx.beginPath();
  const {x_dn, y_dn} = deNormPoints(x, y, rect.width, rect.height);
  ctx.moveTo(x_dn, y_dn);

  points.push({ x, y });
}

function draw(e) {
  if (!isDrawing || hasDrawnLine) return;

  const { x, y } = getPos(e);

  if (lastPoint) {
    const distance = Math.sqrt((lastPoint.x - x)**2 + (lastPoint.y - y)**2);
    if (distance < 0.03) return;
  }

  lastPoint = { x: x, y: y };

  const rect = canvas.getBoundingClientRect();

  const {x_dn, y_dn} = deNormPoints(x, y, rect.width, rect.height);
  ctx.lineTo(x_dn, y_dn);
  ctx.stroke();

  points.push({ x, y });
}

function endDraw() {
  if (!isDrawing) return;

  isDrawing = false;
  hasDrawnLine = true;
  renderPoints();
}

function renderPoints() {
  //const rect = canvas.getBoundingClientRect();
  //const pointsDeNorm = points.map(({x, y}) => deNormPoints(x, y, rect.right - rect.left, rect.top - rect.bottom));
  //pointsOutput.innerHTML = `<pre>${JSON.stringify(pointsDeNorm, null, 2)}</pre>`;
  pointsOutput.innerHTML = `<pre>${JSON.stringify(points, null, 2)}</pre>`;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hasDrawnLine = false;
  points = [];
  pointsOutput.innerHTML = "";
}

ctx.lineWidth = 2;
ctx.lineCap = "round";
ctx.strokeStyle = "#000000";

// Mouse
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);

// Touch
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", clearCanvas);