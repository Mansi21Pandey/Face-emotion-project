// =====================================
// STATIC IMAGE PAGE
// =====================================

const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");

if (analyzeBtn) {
  analyzeBtn.addEventListener("click", async () => {
    if (!imageInput.files.length) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");

    try {
      const res = await fetch("/predict_image", {
        method: "POST",
        body: formData,
      });

      const responseText = await res.text();

      console.log("Predict image HTTP status:", res.status);
      console.log("Predict image response:", responseText);

      if (!res.ok) {
        throw new Error(
          `Server error (${res.status}): ${responseText || "Empty response"}`,
        );
      }

      let data;
      try {
        data = JSON.parse(responseText);
      } catch {
        throw new Error("Server returned an invalid or empty response.");
      }

      if (data.error) {
        alert("Error: " + data.error);
        return;
      }

      document.getElementById("annotatedImg").src = data.annotated_image;
      document.getElementById("orientationTag").textContent =
        `(orientation: ${data.orientation})`;

      renderCharts(data.faces);
      document.getElementById("results").classList.remove("hidden");
    } catch (err) {
      console.error("Image analysis failed:", err);
      alert("Failed: " + err.message);
    } finally {
      document.getElementById("loading").classList.add("hidden");
    }
  });
}

function renderCharts(faces) {
  const container = document.getElementById("chartsContainer");
  container.innerHTML = "";

  if (!faces.length) {
    container.innerHTML = "<p>No faces detected.</p>";
    return;
  }

  faces.forEach((face) => {
    const box = document.createElement("div");
    box.className = "chart-box";
    box.innerHTML = `
      <h3>Face ${face.face_number} — ${face.emotion} (${face.confidence}%)</h3>
      <canvas id="chart-${face.face_number}"></canvas>
    `;
    container.appendChild(box);

    const labels = Object.keys(face.scores);
    const values = Object.values(face.scores);

    new Chart(document.getElementById(`chart-${face.face_number}`), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Confidence (%)",
            data: values,
            backgroundColor: "rgba(0, 200, 150, 0.7)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        scales: { y: { beginAtZero: true, max: 100 } },
        plugins: { legend: { display: false } },
      },
    });
  });
}

// =====================================
// REAL-TIME PAGE
// =====================================

const video = document.getElementById("video");
const output = document.getElementById("output");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const status = document.getElementById("status");

if (startBtn) {
  const socket = io();
  let stream = null;
  let sendInterval = null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  socket.on("annotated_frame", (data) => {
    output.src = data.image;
  });

  startBtn.addEventListener("click", async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      });
      video.srcObject = stream;
      await video.play();

      status.textContent = "Streaming...";
      startBtn.disabled = true;
      stopBtn.disabled = false;

      // Send ~10 frames/sec
      sendInterval = setInterval(() => {
        if (video.videoWidth === 0) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
        socket.emit("frame", { image: dataUrl });
      }, 100);
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  });

  stopBtn.addEventListener("click", () => {
    clearInterval(sendInterval);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    status.textContent = "Stopped";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    output.src = "";
  });
}
