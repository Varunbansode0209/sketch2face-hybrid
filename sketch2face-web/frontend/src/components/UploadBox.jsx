import { useState } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

const UploadBox = ({ onFileSelect }) => {
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      onFileSelect(file);
    } else {
      alert('Please upload an image file');
    }
  };

  const clearImage = () => {
    setPreview(null);
    onFileSelect(null);
  };

  return (
    <div className="w-full">
      {!preview ? (
        <div
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition ${
            dragActive
              ? 'border-blue-600 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400 bg-white'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('fileInput').click()}
        >
          <Upload className="w-16 h-16 mx-auto mb-4 text-blue-500" />
          <p className="text-lg font-semibold text-gray-700 mb-2">
            Upload Forensic Sketch or Photo
          </p>
          <p className="text-sm text-gray-500">
            Drag and drop or click to browse (JPG, PNG, etc.)
          </p>
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />
        </div>
      ) : (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="w-full h-96 object-cover rounded-lg"
          />
          <button
            onClick={clearImage}
            className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadBox;