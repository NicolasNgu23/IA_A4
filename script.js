const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.arc(e.clientX - rect.left, e.clientY - rect.top, 10, 0, Math.PI * 2);
  ctx.fill();
}

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("prediction").innerText = "Prédiction : ...";
}

async function runInference() {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.drawImage(canvas, 0, 0, 28, 28);
  const imgData = tempCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;

  console.log("Image data:", data);

  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4];
    input[i] = r / 255.0;
  }

  console.log(input)
  const tensor = new ort.Tensor("float32", input, [1, 1, 28, 28]);
  const session = await ort.InferenceSession.create("cnn.onnx");
  const feeds = { input: tensor };
  const results = await session.run(feeds);
  const output = results.output.data;
  console.log("Output:", output);

  const predictedClass = output.indexOf(Math.max(...output));
  document.getElementById("prediction").innerText = "Prédiction : " + predictedClass;
}
