import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import { FaUpload, FaRedo, FaCopy, FaDownload, FaFile } from "react-icons/fa";
import { FaArrowRightLong } from "react-icons/fa6";
import styled from "styled-components";
import axios from "axios";

const StyledWrapper = styled.div`
  button {
    margin-top: 10px;
    padding: 1.3em 3em;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 600;
    color: #000;
    background-color: #fff;
    border: none;
    border-radius: 45px;
    box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease 0s;
    cursor: pointer;
    outline: none;
  }

  button:hover {
    background-color: #493d9e;
    box-shadow: 0px 15px 20px rgb(73, 61, 158);
    color: #fff;
    transform: translateY(-7px);
  }

  button:active {
    transform: translateY(-1px);
  }
`;

function App() {
  const [image, setImage] = useState(null);
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    setImage(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png']
    }
  });

  const handleStartOver = () => {
    setImage(null);
    setText("");
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(text);
  };

  const handleDownloadText = () => {
    const element = document.createElement("a");
    const file = new Blob([text], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "extracted_text.txt";
    document.body.appendChild(element);
    element.click();
  };

  const handleConvert = async () => {
    if (image) {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("file", image);

      try {
        const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        setText(response.data.extracted_text);
      } catch (error) {
        console.error("Error uploading image:", error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <div className="flex flex-col items-center flex-grow mt-8 px-4 md:px-0">
        <h1 className="text-3xl md:text-5xl font-medium m-10 font-serif text-center">
          Image to Text Converter
        </h1>
        {!image && (
          <>
            <div
              {...getRootProps()}
              className="firstcontent w-full md:w-190 h-64 border-3 border-dashed border-gray-300 flex items-center justify-center mb-0 p-4 md:p-16 rounded-2xl cursor-pointer"
            >
              <input {...getInputProps()} />
              <div className="insidebox flex flex-col items-center">
                <img className="h-10 w-10 md:h-15 md:w-15" src="src/assets/upload.png" alt="" />
                <p className="text-center text-lg md:text-xl text-gray-800">
                  Drag & drop an image here, or click to select one
                </p>
                <p className="text-gray-500 text-sm md:text-base">
                  Supported formats: JPG, PNG, JPEG
                </p>
                <div className="relative flex items-center px-4 py-2 md:px-8 md:py-3 mt-2 overflow-hidden font-medium transition-all bg-[#493D9E] rounded-md group">
                  <span className="absolute top-0 right-0 inline-block w-4 h-4 transition-all duration-500 ease-in-out bg-[#B2A5FF] rounded group-hover:-mr-4 group-hover:-mt-4">
                    <span className="absolute top-0 right-0 w-5 h-5 rotate-45 translate-x-1/2 -translate-y-1/2 bg-white"></span>
                  </span>
                  <span className="absolute bottom-0 rotate-180 left-0 inline-block w-4 h-4 transition-all duration-500 ease-in-out bg-[#B2A5FF] rounded group-hover:-ml-4 group-hover:-mb-4">
                    <span className="absolute top-0 right-0 w-5 h-5 rotate-45 translate-x-1/2 -translate-y-1/2 bg-white"></span>
                  </span>
                  <span className="absolute bottom-0 left-0 w-full h-full transition-all duration-300 ease-in-out delay-200 -translate-x-full bg-[#DAD2FF] rounded-md group-hover:translate-x-0"></span>
                  <span className="relative w-full flex text-left text-white transition-colors duration-200 ease-in-out group-hover:text-[#493D9E]">
                    <FaFile className="mr-2 mt-1" /> Browse
                  </span>
                </div>
              </div>
            </div>
            <p className="md:pr-75 text-gray-600 text-center mt-4 md:mt-0 ">
              *Your privacy is protected! No data is transmitted or stored.
            </p>
          </>
        )}
        {image && (
          <div className="flex flex-col items-center mt-8">
            <div className="flex flex-col items-center mb-4">
              <img
                src={URL.createObjectURL(image)}
                alt="Uploaded"
                className="w-32 h-32 md:w-48 md:h-48 object-cover rounded-lg shadow-md"
              />
              <p className="mt-2 text-gray-600 text-center">{image.name}</p>
            </div>
            <StyledWrapper>
              <button className="px-4 flex " onClick={handleConvert}>
                <FaUpload className="mr-2" /> Convert to Text
              </button>
            </StyledWrapper>
          </div>
        )}
        {text && (
          <div className="flex flex-col items-center mt-8 w-full px-6">
            <div className="flex flex-col md:flex-row justify-between w-full mb-4">
              <button
                onClick={handleStartOver}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-700 flex items-center mb-4 md:mb-0"
              >
                <FaRedo className="mr-2" /> Start Over
              </button>
              <div className="flex space-x-4">
                <button
                  onClick={handleCopyText}
                  className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-700 flex items-center"
                >
                  <FaCopy className="mr-2" /> Copy Text
                </button>
                <button
                  onClick={handleDownloadText}
                  className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-700 flex items-center"
                >
                  <FaDownload className="mr-2" /> Download Text
                </button>
              </div>
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full h-64 p-4 border border-gray-400 rounded-lg shadow-md bg-white text-black"
            />
          </div>
        )}
        <div className="secondcontent m-5 py-8 px-4 md:px-24 bg-[#DAD2FF] rounded-lg shadow-md flex flex-col md:flex-row">
          <div>
            <h2 className="text-lg font-semibold mb-2">
              Text Extraction Process Overview
            </h2>
            <ul className="list-disc list-inside text-lg text-gray-700">
              <li>
                The image undergoes preprocessing, including grayscale
                conversion, noise reduction, and thresholding.
              </li>
              <li>
                The grayscale image is analyzed using various text extraction
                techniques.
              </li>
              <li>
                Adaptive thresholding ensures text extraction across different
                dimensions.
              </li>
              <li>Extracted text is stored in a structured 2D list.</li>
              <li>A similarity-based analysis filters meaningful words.</li>
              <li>
                Spelling and contextual relevance are verified based on previous
                words.
              </li>
              <li>
                The system returns the most accurate text based on the highest
                similarity score.
              </li>
            </ul>
          </div>
          <img
            className="h-40 w-40 md:h-60 md:w-60 mt-4 md:mt-0"
            src="src/assets/Image upload-rafiki.png"
            alt="Image"
          />
        </div>

        <div className="thirdcontent w-full my-16">
          <div className="flex flex-col md:flex-row justify-center items-center gap-8">
            <div className="flex flex-col items-center">
              <img
                src="src/assets/file.png"
                alt=""
                className="h-20 w-20 md:h-30 md:w-30 mb-2"
              />
              <p className="text-gray-600 text-center">Image Upload</p>
            </div>
            <FaArrowRightLong className="text-xl text-gray-400 hidden md:block" />
            <div className="flex flex-col items-center">
              <img
                src="src/assets/image-correction.png"
                alt=""
                className="h-20 w-20 md:h-30 md:w-30 mb-2"
              />
              <p className="text-gray-600 text-center">
                Image Correction grayscaling
              </p>
            </div>
            <FaArrowRightLong className="text-xl text-gray-400 hidden md:block" />
            <div className="flex flex-col items-center">
              <img
                src="src/assets/image-processing.png"
                alt=""
                className="h-20 w-20 md:h-30 md:w-30 mb-2"
              />
              <p className="text-gray-600 text-center">
                Scanning Image Text extraction
              </p>
            </div>
            <FaArrowRightLong className="text-xl text-gray-400 hidden md:block" />
            <div className="flex flex-col items-center">
              <img
                src="src/assets/file-download.png"
                alt=""
                className="h-20 w-20 md:h-30 md:w-30 mb-2"
              />
              <p className="text-gray-600 text-center">
                Text File Created (Download)
              </p>
            </div>
          </div>
        </div>

        <div className="how-it-works m-5 md:m-20 p-5 md:p-10 bg-[#FFF2AF] rounded-lg shadow-md">
          <h2 className="text-xl md:text-2xl font-semibold mb-6 text-gray-800">
            How does Image to Text Converter work?
          </h2>
          <p className="text-gray-600 mb-4">
            You don't have to do much to copy text from an image if you don't
            know how to convert a JPEG or PNG to text. Simply follow these
            steps:
          </p>
          <div className="steps space-y-4 ml-3">
            <div className="flex items-center gap-3">
              <div className="md:w-8 w-14 h-8 rounded-full bg-[#578FCA] flex items-center justify-center text-white font-bold">
                1
              </div>
              <p className="text-gray-700">
                Upload your image or drag & drop it into the designated area
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="md:w-8 w-14 h-8 rounded-full bg-[#578FCA] flex items-center justify-center text-white font-bold">
                2
              </div>
              <p className="text-gray-700">
                Click the "Convert to Text" button and wait for processing
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="md:w-8 w-8 h-8 rounded-full bg-[#578FCA] flex items-center justify-center text-white font-bold">
                3
              </div>
              <p className="text-gray-700">
                Edit the extracted text if needed</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="md:w-8 w-14 h-8 rounded-full bg-[#578FCA] flex items-center justify-center text-white font-bold">
                4
              </div>
              <p className="text-gray-700">
                Copy the text to your clipboard or save it as a document
              </p>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default App;
