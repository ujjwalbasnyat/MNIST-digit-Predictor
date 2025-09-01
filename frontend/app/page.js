'use client'
import { useRef, useState, useEffect } from 'react'

export default function Home() {
  const description = "This is a machine learning model trained on the famous MNIST dataset. Draw a digit in the box above and click 'Predict' to see the model's prediction!"
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null)
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 25;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#ffffffff';
  }, []);

  const startDrawing = (e) => {
    setDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(
      (e.touches ? e.touches[0].clientX : e.nativeEvent.offsetX),
      (e.touches ? e.touches[0].clientY : e.nativeEvent.offsetY)
    );
  };

  const draw = (e) => {
    if (!drawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let x, y;
    if (e.touches) {
      const rect = canvas.getBoundingClientRect();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.nativeEvent.offsetX;
      y = e.nativeEvent.offsetY;
    }
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setPrediction(null)
    setPreview(null)
  };


  // resizing the canvas
  const preprocessCanvas =(canvas) => {
    const targetSize = 32;
    const offCanvas = document.createElement("canvas")
    offCanvas.width = targetSize;
    offCanvas.height = targetSize;
    const ctx = offCanvas.getContext('2d')

    ctx.drawImage(canvas, 0, 0, targetSize, targetSize);

    return new Promise((resolve) =>
    offCanvas.toBlob((blob)=>resolve(blob), "image/png")
    )
  }

  const sendCanvas = async() => {
    const canvas = canvasRef.current;
    const blob = await preprocessCanvas(canvas);

    // Preview process image
    setPreview(URL.createObjectURL(blob));

    const formData = new FormData()
    formData.append("file", blob, "digit.png");

    const res = await fetch("https://mnist-digit-predictor-pj03.onrender.com/predict", {method : "POST", body: formData, credentials: "include",});

    const result = await res.json();
    setPrediction(result)
  };



  return (
    <div className='bg-[#2C3E50] items-center flex flex-col p-10 min-h-screen'>
      <h1 className='text-white font-bold text-3xl'>Digit Predictor</h1>
      <p className='text-white font-thin pt-2'>Draw a digit & get prediction</p>
      <div className="bg-[#F4F6F9] flex flex-col items-center p-6 mt-4 rounded-lg">
            <button
        className="text-[#2C3E50] bg-white px-4 py-2 rounded mb-4 cursor-pointer"
        onClick={clearCanvas}
      >
        Clear
      </button>
      <div className="text[#2C3E50]">Your drawing canvas</div>
      <canvas
        ref={canvasRef}
        width={300}
        height={300}
        className="border border-[#D8D8D8] rounded bg-[#000000] touch-none"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />
    <button className='text-white bg-[#2C3E50] px-4 py-2 rounded mt-4 cursor-pointer ' onClick={sendCanvas}>Predict</button>

    {prediction && (
      <div>
        <div className='text-[#2C3E50] font-bold mt-4'>The model predicts the digit is: {prediction.digit}</div>
        {prediction.confidence && (
          <div className='text[#2C3E50] font-'>Confidence: {(prediction.confidence*100).toFixed(2)}%</div>
        )}
      </div>
      
    )}
    <div className=' w-full lg:w-[60%] text-center mt-6 text-[#2C3E50] font-thin'>{description}</div>
    {preview && (
      <img src={preview} alt='Canvas preview' width={300} height={300}/>
    )} 
    </div>
    
    </div>
  );
}